#!/usr/bin/env python3
# calib_fit_model.py
"""
Step 3 of the box calibration: learn how large the NM amplitude has to be,
from cheap structure/NMA descriptors, and test it on held-out proteins.

For every protein p and axis x, y, z:
    required_a(p) = MD extent statistic (calib_required_box.py)
    env_a(p, A)   = NM envelope extent at amplitude A (calib_nm_envelope.py)
    A*_a(p)       = smallest A with env_a(p, A) >= required_a(p)
    A*(p)         = max over axes           <- the quantity we model

Model:  log(A* + eps) = b0 + sum_j b_j * feature_j     (quantile regression)

Quantile regression at quantile q predicts an amplitude that is large enough
for a fraction q of proteins, so the box is "safe for ~q of proteins" while
still adapting to size and flexibility. q = 0.95 by default.

Protocol (no information from test proteins is used before the final test):
  1. Random train/test split of proteins (default 25 % test).
  2. On train only: every feature subset of size <= --max-features is
     scored by repeated 5-fold CV. Quantile regression on few proteins
     under-covers, so the out-of-fold residuals give a conformal shift of
     the intercept that restores coverage q (conformalised quantile
     regression, Romano et al. 2019). Subsets are then compared on cost =
     median box volume relative to the standard cubic box.
  3. Pick the cheapest subset (a larger subset must win by --min-gain).
  4. Refit on all train proteins, add the CV shift, and evaluate once on the
     test proteins against the baselines below. The test coverage is the
     honest number to report.

Baselines
  standard_cubic : editconf -bt cubic -d D  (L = max diameter + 2D)
  standard_rect  : editconf -bt triclinic -d D (L_a = extent_a + 2D)
  fixed_A        : one amplitude for everyone (train q-quantile of A*)
  model          : feature-dependent amplitude (this script)

A box side L_a is counted safe when L_a >= required_a + cutoff.
NM boxes use L_a = env_a(A) + cutoff.

Example
  python calib_fit_model.py --pdb-list passing_pdbs.txt \
      --frame lab --stat max --scaling thermal --n-modes 10 \
      --cutoff 1.2 --standard-d 1.0 --quantile 0.95 --tag lab_thermal_K10
"""
from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from calib_geometry import solve_amplitude, envelope_at

EPS = 0.01          # nm, added before log so A* = 0 is allowed

# name -> (source key, transform). Sources: features.json + nma_features.json
FEATURES = {
    "log_n_heavy": ("n_heavy_atoms", "log"),
    "log_rg": ("rg_nm", "log"),
    "asphericity": ("asphericity", None),
    "log_axis_ratio_1_3": ("axis_ratio_1_3", "log"),
    "terminal_coil_max": ("terminal_coil_max", None),
    "coil_fraction": ("coil_fraction", None),
    "log_nma_msf_10": ("nma_msf_10", "log"),
    "log_inv_lambda7": ("inv_lambda7", "log"),
    "mode7_localisation": ("mode7_localisation", None),
}
DEFAULT_CANDIDATES = list(FEATURES)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_pdb_list(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s.split()[0].lower())
    return out


def load_protein(calib: Path, frame: str, stat: str, scaling: str, k: int) -> dict:
    req = json.loads((calib / "required_box.json").read_text())
    env = np.load(calib / "envelope.npz")
    feats = json.loads((calib / "features.json").read_text())
    nfp = calib / "nma_features.json"
    if nfp.exists():
        feats.update(json.loads(nfp.read_text()))

    key = f"{scaling}_K{k}"
    if key not in env:
        raise KeyError(f"{calib}/envelope.npz has no '{key}'. Available: {list(env.keys())}")

    return {
        "required": np.asarray(req["pooled"][frame][f"{stat}_nm"], dtype=float),
        "grid": env["grid"].astype(float),
        "env": env[key].astype(float),          # (n_grid, 3)
        "features": feats,
        "md_box": np.asarray(req["md_box_lengths_nm"], dtype=float),
    }


def build_dataset(root: Path, pdbs: list[str], args) -> tuple[pd.DataFrame, dict]:
    rows, curves = [], {}
    for pdb in pdbs:
        calib = root / pdb / "calib"
        try:
            p = load_protein(calib, args.frame, args.stat, args.scaling, args.n_modes)
        except FileNotFoundError as exc:
            print(f"[skip] {pdb}: {exc}")
            continue
        a_axes = [solve_amplitude(p["grid"], p["env"][:, a], p["required"][a]) for a in range(3)]
        row = {
            "pdb": pdb,
            "A_star": max(a_axes),
            "A_x": a_axes[0], "A_y": a_axes[1], "A_z": a_axes[2],
            "req_x": p["required"][0], "req_y": p["required"][1], "req_z": p["required"][2],
            "extrapolated": max(a_axes) > p["grid"][-1],
        }
        f = p["features"]
        for name, (src, tf) in FEATURES.items():
            v = f.get(src, np.nan)
            v = np.nan if v is None else float(v)
            row[name] = np.log(v) if (tf == "log" and v > 0) else (np.nan if tf == "log" else v)
        for src in ("max_diameter_nm", "extent0_x_nm", "extent0_y_nm", "extent0_z_nm"):
            row[src] = f.get(src, np.nan)
        rows.append(row)
        curves[pdb] = p
    df = pd.DataFrame(rows)
    bad = ~np.isfinite(df["A_star"])
    if bad.any():
        print(f"[warn] dropping {bad.sum()} proteins whose envelope never reaches the requirement: "
              f"{df.loc[bad, 'pdb'].tolist()}")
        df = df[~bad]
    if df["extrapolated"].any():
        print(f"[warn] {df['extrapolated'].sum()} proteins needed extrapolation beyond the amplitude grid; "
              "consider a larger --amp-max in calib_nm_envelope.py")
    return df.reset_index(drop=True), curves


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class QuantileModel:
    """Standardised linear quantile regression on log(A* + EPS)."""

    def __init__(self, features: list[str], q: float, l1: float = 0.0):
        self.features = list(features)
        self.q = q
        self.l1 = l1
        self.shift = 0.0          # conformal correction on the log scale

    def fit(self, df: pd.DataFrame) -> "QuantileModel":
        y = np.log(df["A_star"].to_numpy() + EPS)
        if not self.features:
            self.mu = np.zeros(0)
            self.sd = np.ones(0)
            self.coef = np.zeros(0)
            self.intercept = float(np.quantile(y, self.q))
            return self
        from sklearn.linear_model import QuantileRegressor
        X = df[self.features].to_numpy(dtype=float)
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qr = QuantileRegressor(quantile=self.q, alpha=self.l1, solver="highs").fit(Z, y)
        self.coef = qr.coef_.astype(float)
        self.intercept = float(qr.intercept_)
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        if self.features:
            Z = (df[self.features].to_numpy(dtype=float) - self.mu) / self.sd
            return self.intercept + Z @ self.coef
        return np.full(len(df), self.intercept)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        y = self.predict_log(df) + self.shift
        return np.maximum(np.exp(y) - EPS, 0.0)

    def to_dict(self) -> dict:
        return {
            "target": "log(A_star + EPS), A in nm RMS per-atom displacement",
            "EPS": EPS,
            "quantile": self.q,
            "features": self.features,
            "feature_definitions": {f: FEATURES[f] for f in self.features},
            "standardisation_mean": self.mu.tolist(),
            "standardisation_sd": self.sd.tolist(),
            "coef_standardised": self.coef.tolist(),
            "intercept": self.intercept,
            "conformal_shift": self.shift,
            "predict": "A = max(exp(intercept + conformal_shift + "
                       "sum(coef * (x - mean) / sd)) - EPS, 0)",
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def nm_box(p: dict, A: float, cutoff: float) -> np.ndarray:
    ext = np.array([envelope_at(p["grid"], p["env"][:, a], A) for a in range(3)])
    return ext + cutoff


def evaluate(df: pd.DataFrame, curves: dict, amps: np.ndarray, args, method: str) -> pd.DataFrame:
    out = []
    for (_, r), A in zip(df.iterrows(), amps):
        p = curves[r["pdb"]]
        need = p["required"] + args.cutoff
        if method == "standard_cubic":
            box = np.full(3, r["max_diameter_nm"] + 2 * args.standard_d)
        elif method == "standard_rect":
            box = np.array([r["extent0_x_nm"], r["extent0_y_nm"], r["extent0_z_nm"]]) + 2 * args.standard_d
        else:
            box = nm_box(p, A, args.cutoff)
        cubic = (r["max_diameter_nm"] + 2 * args.standard_d) ** 3
        out.append({
            "pdb": r["pdb"], "method": method,
            "A_pred": A if method not in ("standard_cubic", "standard_rect") else np.nan,
            "A_star": r["A_star"],
            "Lx": box[0], "Ly": box[1], "Lz": box[2],
            "safe": bool(np.all(box >= need - 1e-9)),
            "worst_shortfall_nm": float(np.max(need - box)),
            "volume_nm3": float(np.prod(box)),
            "volume_vs_cubic": float(np.prod(box) / cubic),
        })
    return pd.DataFrame(out)


def summarise(ev: pd.DataFrame) -> dict:
    return {
        "n": int(len(ev)),
        "coverage": float(ev["safe"].mean()),
        "median_volume_vs_cubic": float(ev["volume_vs_cubic"].median()),
        "mean_volume_vs_cubic": float(ev["volume_vs_cubic"].mean()),
        "p90_volume_vs_cubic": float(ev["volume_vs_cubic"].quantile(0.9)),
        "worst_shortfall_nm": float(ev["worst_shortfall_nm"].max()),
    }


def conformal_quantile(res: np.ndarray, q: float, n_eff: int) -> float:
    """Finite-sample conformal quantile of residuals: level ceil((n+1)q)/n."""
    level = min(1.0, np.ceil((n_eff + 1) * q) / n_eff)
    return float(np.quantile(res, level, method="higher"))


def cv_score(train: pd.DataFrame, curves: dict, feats: list[str], args, rng_seed: int) -> dict:
    """
    Repeated K-fold CV on the training proteins.

    Quantile regression fitted on few points under-covers on new data, so we
    use conformalised quantile regression (Romano et al., NeurIPS 2019):
    out-of-fold residuals r = log(A*+EPS) - prediction give a shift that
    brings coverage up to q. All candidate feature sets are then compared at
    the same coverage, on box volume.
    """
    from sklearn.model_selection import RepeatedKFold
    n = len(train)
    n_splits = min(5, n)
    y = np.log(train["A_star"].to_numpy() + EPS)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=args.cv_repeats, random_state=rng_seed)
    oof = np.full((args.cv_repeats, n), np.nan)
    for i, (tr_idx, va_idx) in enumerate(rkf.split(train)):
        m = QuantileModel(feats, args.quantile, args.l1).fit(train.iloc[tr_idx])
        oof[i // n_splits, va_idx] = m.predict_log(train.iloc[va_idx])
    res = (y[None, :] - oof)                       # > 0 means box too small
    raw_cov = float((res <= 0).mean())
    shift = conformal_quantile(res.ravel(), args.quantile, n) if args.conformal else 0.0

    # cost and coverage of the shifted out-of-fold predictions
    evs = []
    for i in range(args.cv_repeats):
        A = np.maximum(np.exp(oof[i] + shift) - EPS, 0.0)
        evs.append(evaluate(train, curves, A, args, "model"))
    s = summarise(pd.concat(evs))
    s["raw_coverage_before_shift"] = raw_cov
    s["conformal_shift"] = shift
    s["features"] = "+".join(feats) if feats else "(intercept only)"
    s["n_features"] = len(feats)
    return s


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(outdir: Path, df: pd.DataFrame, test_ev: pd.DataFrame, feats: list[str], q: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = feats or ["log_n_heavy", "log_rg"]
    show = [f for f in show if f in df][:4]
    if show:
        fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 3.6), squeeze=False)
        for ax, f in zip(axes[0], show):
            ax.scatter(df[f], df["A_star"], s=14, alpha=0.7)
            ax.set_xlabel(f)
            ax.set_ylabel("A* (nm RMS)")
            ax.set_yscale("symlog", linthresh=0.05)
        fig.tight_layout()
        fig.savefig(outdir / "A_star_vs_features.png", dpi=200)
        plt.close(fig)

    m = test_ev[test_ev["method"] == "model"]
    if len(m):
        fig, ax = plt.subplots(figsize=(4.2, 4))
        c = np.where(m["safe"], "tab:blue", "tab:red")
        ax.scatter(m["A_star"], m["A_pred"], c=c, s=16)
        lim = [0, max(m["A_star"].max(), m["A_pred"].max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("A* needed (nm RMS)")
        ax.set_ylabel(f"A predicted (q={q})")
        ax.set_title("Test proteins (red = box too small)")
        fig.tight_layout()
        fig.savefig(outdir / "test_predicted_vs_needed.png", dpi=200)
        plt.close(fig)

    methods = list(test_ev["method"].unique())
    fig, ax = plt.subplots(figsize=(1.6 * len(methods) + 2, 4))
    data = [test_ev.loc[test_ev["method"] == mm, "volume_vs_cubic"] for mm in methods]
    ax.boxplot(data, showfliers=True)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([f"{mm}\ncov={test_ev.loc[test_ev['method'] == mm, 'safe'].mean():.0%}"
                        for mm in methods], fontsize=8)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("box volume / standard cubic")
    ax.set_title("Held-out proteins")
    fig.tight_layout()
    fig.savefig(outdir / "test_methods_volume.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb-list", type=Path, help="Text file, one PDB code per line")
    src.add_argument("--pdbs", nargs="+")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--frame", choices=["lab", "body"], default="lab",
                    help="lab includes tumbling (plain MD); body = rotation removed")
    ap.add_argument("--stat", choices=["max", "p999", "p99"], default="max")
    ap.add_argument("--scaling", choices=["uniform", "thermal"], default="thermal")
    ap.add_argument("--n-modes", type=int, default=10)
    ap.add_argument("--cutoff", type=float, default=1.2,
                    help="Minimum allowed periodic-image distance (nm)")
    ap.add_argument("--standard-d", type=float, default=1.0,
                    help="Solute-box distance of the standard editconf rule (nm)")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--target-coverage", type=float, default=None,
                    help="Required CV coverage for feature selection (default = quantile)")
    ap.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES,
                    choices=list(FEATURES))
    ap.add_argument("--max-features", type=int, default=2)
    ap.add_argument("--l1", type=float, default=0.0, help="L1 penalty of QuantileRegressor")
    ap.add_argument("--no-conformal", dest="conformal", action="store_false",
                    help="Disable the conformal coverage correction (not recommended)")
    ap.add_argument("--min-gain", type=float, default=0.02,
                    help="Relative volume gain needed to prefer a larger feature set")
    ap.add_argument("--test-fraction", type=float, default=0.25)
    ap.add_argument("--cv-repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    target_cov = args.target_coverage if args.target_coverage is not None else args.quantile

    pdbs = read_pdb_list(args.pdb_list) if args.pdb_list else [p.lower() for p in args.pdbs]
    tag = args.tag or f"{args.frame}_{args.stat}_{args.scaling}_K{args.n_modes}_q{args.quantile}"
    outdir = args.results / "calibration" / tag
    outdir.mkdir(parents=True, exist_ok=True)

    df, curves = build_dataset(args.results, pdbs, args)
    n = len(df)
    print(f"[data] {n} proteins usable")
    if n < 20:
        print("[warn] fewer than 20 proteins: coverage estimates will be very noisy; "
              "keep --max-features at 1")
    df.to_csv(outdir / "dataset.csv", index=False)

    usable = [c for c in args.candidates if df[c].notna().all()]
    for c in set(args.candidates) - set(usable):
        print(f"[warn] feature {c} has missing values; not used")

    # 1. split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(args.test_fraction * n)))
    test = df.iloc[perm[:n_test]].reset_index(drop=True)
    train = df.iloc[perm[n_test:]].reset_index(drop=True)
    print(f"[split] train={len(train)}  test={len(test)}")

    # 2. feature search on train
    rows = []
    for k in range(0, args.max_features + 1):
        for subset in itertools.combinations(usable, k):
            s = cv_score(train, curves, list(subset), args, args.seed)
            rows.append(s)
            print(f"  cov={s['coverage']:.3f}  vol/cubic={s['median_volume_vs_cubic']:.3f}  {s['features']}")
    search = pd.DataFrame(rows).sort_values(["n_features", "median_volume_vs_cubic"])
    search.to_csv(outdir / "feature_search_cv.csv", index=False)

    # 3. choose: among subsets reaching the target coverage, the smallest
    #    median volume; a bigger subset must beat a smaller one by more than
    #    --min-gain (relative) to be preferred.
    ok = search[search["coverage"] >= target_cov - 1e-9]
    if ok.empty:
        print(f"[warn] no subset reaches CV coverage {target_cov}; "
              "using all subsets (use --conformal or a lower --quantile)")
        ok = search
    best_vol = ok["median_volume_vs_cubic"].min()
    within = ok[ok["median_volume_vs_cubic"] <= best_vol * (1 + args.min_gain)]
    best = within.sort_values(["n_features", "median_volume_vs_cubic"]).iloc[0]
    feats = [] if best["features"] == "(intercept only)" else best["features"].split("+")
    print(f"[select] {best['features']}  (CV coverage {best['coverage']:.3f}, "
          f"median vol/cubic {best['median_volume_vs_cubic']:.3f})")

    # 4. final fit + held-out test (shifts come from the train CV only)
    model = QuantileModel(feats, args.quantile, args.l1).fit(train)
    model.shift = float(best["conformal_shift"])
    fixed = QuantileModel([], args.quantile).fit(train)
    fixed.shift = float(search.loc[search["n_features"] == 0, "conformal_shift"].iloc[0])
    test_ev = pd.concat([
        evaluate(test, curves, np.full(len(test), np.nan), args, "standard_cubic"),
        evaluate(test, curves, np.full(len(test), np.nan), args, "standard_rect"),
        evaluate(test, curves, fixed.predict(test), args, "fixed_A"),
        evaluate(test, curves, model.predict(test), args, "model"),
    ])
    test_ev.to_csv(outdir / "test_predictions.csv", index=False)

    summary = {
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_proteins": n, "n_train": len(train), "n_test": len(test),
        "selected_features": feats,
        "cv_of_selected": best.to_dict(),
        "fixed_A_nm": float(fixed.predict(test.iloc[:1])[0]),
        "test": {m: summarise(g) for m, g in test_ev.groupby("method")},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (outdir / "model.json").write_text(json.dumps(model.to_dict(), indent=2))

    try:
        make_plots(outdir, df, test_ev, feats, args.quantile)
    except Exception as exc:
        print(f"[warn] plotting failed: {exc}")

    print("\nHeld-out test proteins")
    print(f"{'method':16s} {'coverage':>9s} {'median vol/cubic':>17s} {'worst short (nm)':>17s}")
    for m, s in summary["test"].items():
        print(f"{m:16s} {s['coverage']:9.2%} {s['median_volume_vs_cubic']:17.3f} {s['worst_shortfall_nm']:17.3f}")
    print(f"\nwritten to {outdir}")


if __name__ == "__main__":
    main()
