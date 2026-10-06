#!/usr/bin/env python3
# calib_fit_padding.py
"""
Alternative to calib_fit_model.py: learn the padding directly.

Instead of an NM amplitude, the quantity modelled is how far the protein grew
beyond its starting structure in MD:

    growth_a(p) = MD extent statistic along a - starting extent along a
    G(p)        = max over axes                      <- the quantity we model
    box side    L_a = starting extent_a + padding(p) + cutoff

    padding(p) = b0 + sum_j b_j * feature_j          (quantile regression, nm)

The descriptors may still come from NMA (softness of the softest mode, NMA
fluctuation), but the mode shapes are not used to shape the box. Protocol,
train/test split and conformal correction are the same as in
calib_fit_model.py, so the two can be compared row by row (same --seed and
protein list give the same test proteins).

Needs step 1 (calib_required_box.py) and, for NMA descriptors, step 2.

Methods in the final table
  standard_cubic   : editconf -bt cubic -d D
  standard_rect    : editconf -bt triclinic -d D (padding = 2D - cutoff)
  constant_padding : one padding for everyone (train quantile of G)
  model            : descriptor-dependent padding (this script)

Example
  python calib_fit_padding.py --pdb-list passing_pdbs.txt --quantile 0.95
"""
from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from calib_fit_model import FEATURES, DEFAULT_CANDIDATES, read_pdb_list, conformal_quantile


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataset(root: Path, pdbs: list[str], frame: str, stat: str) -> pd.DataFrame:
    rows = []
    for pdb in pdbs:
        calib = root / pdb / "calib"
        try:
            req = json.loads((calib / "required_box.json").read_text())
            f = json.loads((calib / "features.json").read_text())
        except FileNotFoundError as exc:
            print(f"[skip] {pdb}: {exc}")
            continue
        nfp = calib / "nma_features.json"
        if nfp.exists():
            f.update(json.loads(nfp.read_text()))
        need = np.asarray(req["pooled"][frame][f"{stat}_nm"], dtype=float)
        ext0 = np.array([f["extent0_x_nm"], f["extent0_y_nm"], f["extent0_z_nm"]], dtype=float)
        md_box = np.asarray(req["md_box_lengths_nm"], dtype=float)
        g = need - ext0
        row = {
            "pdb": pdb, "G": float(g.max()),
            "g_x": g[0], "g_y": g[1], "g_z": g[2],
            "req_x": need[0], "req_y": need[1], "req_z": need[2],
            "extent0_x_nm": ext0[0], "extent0_y_nm": ext0[1], "extent0_z_nm": ext0[2],
            "max_diameter_nm": float(f["max_diameter_nm"]),
            "md_box_volume_nm3": float(np.prod(md_box)),
        }
        for name, (src, tf) in FEATURES.items():
            v = f.get(src, np.nan)
            v = np.nan if v is None else float(v)
            row[name] = np.log(v) if (tf == "log" and v > 0) else (np.nan if tf == "log" else v)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PaddingModel:
    """Standardised linear quantile regression on the growth G (nm)."""

    def __init__(self, features: list[str], q: float):
        self.features = list(features)
        self.q = q
        self.shift = 0.0

    def fit(self, df: pd.DataFrame) -> "PaddingModel":
        y = df["G"].to_numpy()
        if not self.features:
            self.mu, self.sd, self.coef = np.zeros(0), np.ones(0), np.zeros(0)
            self.intercept = float(np.quantile(y, self.q))
            return self
        from sklearn.linear_model import QuantileRegressor
        X = df[self.features].to_numpy(dtype=float)
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qr = QuantileRegressor(quantile=self.q, alpha=0.0, solver="highs").fit((X - self.mu) / self.sd, y)
        self.coef = qr.coef_.astype(float)
        self.intercept = float(qr.intercept_)
        return self

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        if self.features:
            Z = (df[self.features].to_numpy(dtype=float) - self.mu) / self.sd
            return self.intercept + Z @ self.coef
        return np.full(len(df), self.intercept)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.predict_raw(df) + self.shift, 0.0)

    def to_dict(self) -> dict:
        return {
            "target": "padding in nm: box side = starting extent + padding + cutoff",
            "quantile": self.q,
            "features": self.features,
            "feature_definitions": {f: FEATURES[f] for f in self.features},
            "standardisation_mean": self.mu.tolist(),
            "standardisation_sd": self.sd.tolist(),
            "coef_standardised": self.coef.tolist(),
            "intercept": self.intercept,
            "conformal_shift": self.shift,
            "predict": "padding = max(intercept + conformal_shift + sum(coef * (x - mean) / sd), 0)",
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(df: pd.DataFrame, pad: np.ndarray, cutoff: float, standard_d: float, method: str) -> pd.DataFrame:
    ext0 = df[["extent0_x_nm", "extent0_y_nm", "extent0_z_nm"]].to_numpy()
    need = df[["req_x", "req_y", "req_z"]].to_numpy() + cutoff
    cubic_side = df["max_diameter_nm"].to_numpy() + 2 * standard_d
    if method == "standard_cubic":
        box = np.repeat(cubic_side[:, None], 3, axis=1)
    else:
        box = ext0 + np.asarray(pad)[:, None] + cutoff
    vol = box.prod(1)
    return pd.DataFrame({
        "pdb": df["pdb"].to_numpy(), "method": method,
        "padding_nm": np.nan if method == "standard_cubic" else pad,
        "growth_nm": df["G"].to_numpy(),
        "Lx": box[:, 0], "Ly": box[:, 1], "Lz": box[:, 2],
        "safe": np.all(box >= need - 1e-9, axis=1),
        "worst_shortfall_nm": (need - box).max(1),
        "volume_nm3": vol,
        "volume_vs_cubic": vol / cubic_side ** 3,
        "volume_vs_md_box": vol / df["md_box_volume_nm3"].to_numpy(),
    })


def summarise(ev: pd.DataFrame) -> dict:
    return {
        "n": int(len(ev)),
        "coverage": float(ev["safe"].mean()),
        "median_padding_nm": float(ev["padding_nm"].median()),
        "median_volume_vs_cubic": float(ev["volume_vs_cubic"].median()),
        "median_volume_vs_md_box": float(ev["volume_vs_md_box"].median()),
        "worst_shortfall_nm": float(ev["worst_shortfall_nm"].max()),
    }


def cv_score(train: pd.DataFrame, feats: list[str], args) -> dict:
    """Repeated 5-fold CV with a conformal shift, as in calib_fit_model.cv_score."""
    from sklearn.model_selection import RepeatedKFold
    n = len(train)
    n_splits = min(5, n)
    y = train["G"].to_numpy()
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=args.cv_repeats, random_state=args.seed)
    oof = np.full((args.cv_repeats, n), np.nan)
    for i, (tr, va) in enumerate(rkf.split(train)):
        m = PaddingModel(feats, args.quantile).fit(train.iloc[tr])
        oof[i // n_splits, va] = m.predict_raw(train.iloc[va])
    res = y[None, :] - oof                              # > 0 means padding too small
    shift = conformal_quantile(res.ravel(), args.quantile, n)
    evs = [evaluate(train, np.maximum(oof[i] + shift, 0.0), args.cutoff, args.standard_d, "model")
           for i in range(args.cv_repeats)]
    s = summarise(pd.concat(evs))
    s["raw_coverage_before_shift"] = float((res <= 0).mean())
    s["conformal_shift"] = shift
    s["features"] = "+".join(feats) if feats else "(constant)"
    s["n_features"] = len(feats)
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb-list", type=Path)
    src.add_argument("--pdbs", nargs="+")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--frame", choices=["lab", "body"], default="lab")
    ap.add_argument("--stat", choices=["max", "p999", "p99"], default="max")
    ap.add_argument("--cutoff", type=float, default=1.2)
    ap.add_argument("--standard-d", type=float, default=1.0)
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES, choices=list(FEATURES))
    ap.add_argument("--max-features", type=int, default=2)
    ap.add_argument("--min-gain", type=float, default=0.02,
                    help="Relative volume gain needed to prefer a larger feature set")
    ap.add_argument("--test-fraction", type=float, default=0.25)
    ap.add_argument("--cv-repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    pdbs = read_pdb_list(args.pdb_list) if args.pdb_list else [p.lower() for p in args.pdbs]
    tag = args.tag or f"padding_{args.frame}_{args.stat}_q{args.quantile}"
    outdir = args.results / "calibration" / tag
    outdir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(args.results, pdbs, args.frame, args.stat)
    n = len(df)
    print(f"[data] {n} proteins usable")
    df.to_csv(outdir / "dataset.csv", index=False)
    usable = [c for c in args.candidates if df[c].notna().all()]
    for c in sorted(set(args.candidates) - set(usable)):
        print(f"[warn] feature {c} has missing values; not used")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(args.test_fraction * n)))
    test = df.iloc[perm[:n_test]].reset_index(drop=True)
    train = df.iloc[perm[n_test:]].reset_index(drop=True)
    print(f"[split] train={len(train)}  test={len(test)}")

    rows = []
    for k in range(0, args.max_features + 1):
        for subset in itertools.combinations(usable, k):
            s = cv_score(train, list(subset), args)
            rows.append(s)
            print(f"  cov={s['coverage']:.3f}  pad={s['median_padding_nm']:.2f}  "
                  f"vol/cubic={s['median_volume_vs_cubic']:.3f}  {s['features']}")
    search = pd.DataFrame(rows).sort_values(["n_features", "median_volume_vs_cubic"])
    search.to_csv(outdir / "feature_search_cv.csv", index=False)

    ok = search[search["coverage"] >= args.quantile - 1e-9]
    if ok.empty:
        print(f"[warn] no subset reaches CV coverage {args.quantile}; using all subsets")
        ok = search
    # a rule with more descriptors must beat every simpler one by --min-gain
    best = ok[ok["n_features"] == ok["n_features"].min()].sort_values("median_volume_vs_cubic").iloc[0]
    for k in sorted(ok["n_features"].unique())[1:]:
        cand = ok[ok["n_features"] == k].sort_values("median_volume_vs_cubic").iloc[0]
        if cand["median_volume_vs_cubic"] < best["median_volume_vs_cubic"] * (1 - args.min_gain):
            best = cand
    feats = [] if best["features"] == "(constant)" else best["features"].split("+")
    print(f"[select] {best['features']}  (CV coverage {best['coverage']:.3f}, "
          f"median vol/cubic {best['median_volume_vs_cubic']:.3f})")

    model = PaddingModel(feats, args.quantile).fit(train)
    model.shift = float(best["conformal_shift"])
    const = PaddingModel([], args.quantile).fit(train)
    const.shift = float(search.loc[search["n_features"] == 0, "conformal_shift"].iloc[0])
    std_pad = np.full(len(test), 2 * args.standard_d - args.cutoff)
    test_ev = pd.concat([
        evaluate(test, std_pad, args.cutoff, args.standard_d, "standard_cubic"),
        evaluate(test, std_pad, args.cutoff, args.standard_d, "standard_rect"),
        evaluate(test, const.predict(test), args.cutoff, args.standard_d, "constant_padding"),
        evaluate(test, model.predict(test), args.cutoff, args.standard_d, "model"),
    ])
    test_ev.to_csv(outdir / "test_predictions.csv", index=False)

    summary = {
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_proteins": n, "n_train": len(train), "n_test": len(test),
        "selected_features": feats,
        "cv_of_selected": best.to_dict(),
        "constant_padding_nm": float(const.predict(test.iloc[:1])[0]),
        "test": {m: summarise(g) for m, g in test_ev.groupby("method")},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (outdir / "padding_model.json").write_text(json.dumps(model.to_dict(), indent=2))

    print(f"\nconstant padding: {summary['constant_padding_nm']:.2f} nm")
    print("Held-out test proteins")
    print(f"{'method':17s} {'coverage':>9s} {'median pad':>11s} {'vol/cubic':>10s} "
          f"{'vol/MD box':>11s} {'worst short (nm)':>17s}")
    for m in ("standard_cubic", "standard_rect", "constant_padding", "model"):
        s = summary["test"][m]
        print(f"{m:17s} {s['coverage']:9.2%} {s['median_padding_nm']:11.2f} "
              f"{s['median_volume_vs_cubic']:10.3f} {s['median_volume_vs_md_box']:11.3f} "
              f"{s['worst_shortfall_nm']:17.3f}")
    print(f"\nwritten to {outdir}")


if __name__ == "__main__":
    main()
