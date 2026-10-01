# Box-size calibration: how to run it

This guide covers the three `calib_*` scripts: what they need, how to run
them on the cluster, and what comes out.

What they do: for every protein that has MD, measure how much room it
actually needed, measure how the NM-helper box grows with amplitude, then
learn an amplitude rule from cheap descriptors (size, shape, flexible tails,
NMA flexibility) and test it on proteins the rule has never seen.

```text
            per protein (SLURM array)                       once, all proteins
 ┌──────────────────────────────┐   ┌──────────────────────────────┐   ┌─────────────────────────┐
 │ 1  calib_required_box.py     │──▶│ 2  calib_nm_envelope.py      │──▶│ 3  calib_fit_model.py   │
 │    MD replicas → room needed │   │    NMA → box vs amplitude    │   │    rule + held-out test │
 └──────────────────────────────┘   └──────────────────────────────┘   └─────────────────────────┘
        results/<pdb>/calib/                results/<pdb>/calib/          results/calibration/<tag>/
```

---

## 0. Before you start

### Software

| Needed for | Packages |
|---|---|
| all steps | Python 3.10+, numpy, scipy, pandas, scikit-learn (≥ 1.1), matplotlib |
| step 1 | mdtraj |
| step 2 with `--run-nma` | R with `bio3d` and `Rcpp`, Python `rpy2` (the same setup `main.py` uses) |

This is the same environment as `main.py`. Check it with:

```bash
cd pbc-police                # run everything from the repository root
python test_calib.py         # ~1-2 min, synthetic data only; ends with "all tests passed"
```

### Input data (already on the cluster)

Nothing new has to be prepared. The scripts read the dynamics database
through `io_utils.py`, exactly like `main.py`:

```text
/work001/misc/bekker/kakC/dynamicsdb/raw/<phase>/<pdb>/
├── build/npt.gro                     topology + structure NMA is computed on
└── validation/0..9/prod.partNNNN.xtc MD replicas (heavy atoms, wrapped is fine)
```

The newest phase containing the PDB code is used (`get_pdb_dir`).

### The protein list

A text file with one PDB code per line; `#` lines are ignored. You already
have `passing_pdbs.txt`. Check how many proteins it has:

```bash
grep -cv '^\s*#' passing_pdbs.txt
```

With fewer than about 40 proteins the held-out test set is small (~10) and
its numbers will be noisy (see step 3).

---

## 1. Measure the room each protein needed: `calib_required_box.py`

For every MD frame of every replica, the protein's heavy atoms are made
whole across the periodic boundary (multi-chain complexes are handled), and
the protein's size along x, y and z (max minus min coordinate) is recorded:

* **lab**: as simulated, including the protein tumbling in the box. This
  is what a normal, unrestrained simulation box must hold.
* **body**: after superposing the frame onto the reference (CA atoms), so
  rotation is removed and only flexibility is left.

Run one protein to try it:

```bash
python calib_required_box.py --pdb 1abc
```

| Option | Default | Meaning |
|---|---|---|
| `--pdb` | required | PDB code |
| `--outdir` | `results` | output root; files go to `<outdir>/<pdb>/calib/` |
| `--stride` | `1` | use every n-th frame. Keep 1 for the final run: the maximum can come from a single frame |
| `--chunk` | `500` | frames read at a time (memory) |
| `--align-selection` | `name CA` | atoms used for the body-frame superposition |

Console output, one line per replica (numbers illustrative):

```text
[rep 0] frames=5001  max lab=[6.12 5.87 5.40]  max body=[5.31 4.62 4.05]
...
pooled max extent lab : [6.25 5.93 5.52] nm
MD box                : [8.10 8.10 8.10] nm
```

Output in `results/<pdb>/calib/`:

| File | Content |
|---|---|
| `required_box.json` | pooled and per-replica size along x/y/z for `lab` and `body`: `max_nm`, `p999_nm`, `p99_nm`, `median_nm`, `n_frames`; also the MD box (`md_box_lengths_nm`) and `min_axis_margin_nm` = MD box minus largest size, a quick check against `pidist.py` |
| `extents.npz` | per-frame sizes: arrays `lab`, `body` (n_frames × 3), `replica`, `time_ps` |
| `features.json` | descriptors of the reference: `n_heavy_atoms`, `n_residues`, `n_chains`, `rg_nm`, `asphericity`, `axis_ratio_1_3`, `axis_ratio_2_3`, `max_diameter_nm`, `extent0_x/y/z/max_nm`, `terminal_coil_total`, `terminal_coil_max`, `coil_fraction` |
| `reference_xyz.npy`, `reference.pdb` | the reference structure (frame 0 of the first replica, made whole). Step 2 uses it so both steps share one orientation |
| `nma_reference_xyz.npy` | the `npt.gro` protein coordinates, made whole, which NMA is computed on |

> **Warning to look out for:** `npt.gro protein was broken across PBC`.
> If you see it, earlier NMA results from `main.py` for that protein were
> computed on a broken structure. Step 2 with `--run-nma` uses the repaired
> one.

Runtime is dominated by reading the XTCs: roughly as long as one pass of
`main.py`'s PCA over the same files.

---

## 2. Measure how the NM box grows: `calib_nm_envelope.py`

The reference is pushed by ± a along each of the first K internal modes
(Bio3D modes 7…6+K). For a grid of amplitudes a, the size along x/y/z of
the union of all these structures is stored.

**Amplitude** a is the RMS displacement per atom in nm, so a = 0.2 means the
same thing for a small and a large protein.

* `uniform`: every mode moves by a.
* `thermal`: mode m moves by a·√(λ₇/λₘ), so softer modes move further.

Both scalings and several K are always computed; you pick one in step 3.

```bash
# NMA modes not computed yet: compute them (Bio3D), then the envelope
python calib_nm_envelope.py --pdb 1abc --run-nma

# modes already exist, e.g. from nm_box_size_predictor_per_axis.py
python calib_nm_envelope.py --pdb 1abc --nma-dir boxpred_results/1abc/raw_data/nma
```

| Option | Default | Meaning |
|---|---|---|
| `--run-nma` | off | run Bio3D `aanma` first, saving modes to `--nma-dir` |
| `--nma-dir` | `results/<pdb>/calib/nma` | folder with `raw_modes_all.npy` (3N × modes) and `eigenvalues_all.npy` |
| `--n-modes` | `5 10 20` | the values of K to build envelopes for |
| `--amp-max` | `2.0` | largest amplitude in the grid (nm) |
| `--amp-step` | `0.02` | grid spacing (nm) |

Existing mode files can be reused only if they were computed on the same
atoms in the same order (protein heavy atoms from
`build_protein_heavy_views`). The script stops with an error if the number
of atoms doesn't match.

Console output (numbers illustrative):

```text
[uniform K=10] extent at a=0: [5.10 4.62 4.05]  a=0.50: [6.81 6.20 5.73]
[thermal K=10] extent at a=0: [5.10 4.62 4.05]  a=0.50: [6.02 5.55 5.11]
```

Output in `results/<pdb>/calib/`:

| File | Content |
|---|---|
| `envelope.npz` | `grid` (amplitudes) and one array per setting, e.g. `thermal_K10`, `uniform_K5`, each (n_grid × 3): size along x/y/z at each amplitude |
| `nma_features.json` | `nma_msf_5/10/20` (Σ 1/λ per atom: overall softness), `inv_lambda7` (softest mode), `mode7_localisation` (high = a tail or loop dominates the softest mode) |

---

## Steps 1 + 2 for all proteins: SLURM array

`calib_slurm_example.sh` runs steps 1 and 2 for one protein per array task
and skips any step whose output already exists, so a crashed array can just be
resubmitted. Edit the `#SBATCH` lines and the environment line, then:

```bash
mkdir -p logs
N=$(grep -cv '^\s*#' passing_pdbs.txt)
sbatch --array=1-${N}%20 calib_slurm_example.sh
```

The script uses `--run-nma` unless `results/<pdb>/calib/nma/raw_modes_all.npy`
already exists. To reuse NMA from `boxpred_results/`, change the
`calib_nm_envelope.py` line to pass `--nma-dir`.

Check that everything finished:

```bash
for p in $(grep -v '^\s*#' passing_pdbs.txt); do
  [ -f results/$p/calib/envelope.npz ] || echo "missing: $p"
done
```

---

## 3. Fit and test the amplitude rule: `calib_fit_model.py`

Run once, after steps 1 and 2 are done for all proteins:

```bash
python calib_fit_model.py --pdb-list passing_pdbs.txt \
    --frame lab --stat max --scaling thermal --n-modes 10 \
    --cutoff 1.2 --standard-d 1.0 --quantile 0.95
```

What happens inside:

1. For each protein, **A\*** = the smallest amplitude whose NM box is at
   least as large as the room the protein needed in MD, on all three axes.
2. Proteins are split at random: 75 % train, 25 % test. The test proteins
   are not touched until the last step.
3. On the train proteins, every combination of up to 2 descriptors is
   scored by repeated 5-fold cross-validation. Each candidate rule is a
   quantile regression of log A\* on the descriptors, shifted up just enough
   (conformal correction) that 95 % of proteins get a large enough box.
   Candidates are compared on box volume.
4. The cheapest rule is refit on all train proteins and applied once to the
   test proteins, next to three baselines.

| Option | Default | Meaning |
|---|---|---|
| `--pdb-list` / `--pdbs` | required | proteins to use |
| `--frame` | `lab` | `lab` = box for normal MD (includes tumbling); `body` = flexibility only |
| `--stat` | `max` | which statistic of the MD sizes to cover: `max`, `p999`, `p99` |
| `--scaling` | `thermal` | `thermal` or `uniform` (from step 2) |
| `--n-modes` | `10` | K, must be one computed in step 2 |
| `--cutoff` | `1.2` | minimum allowed distance to a periodic image (nm) |
| `--standard-d` | `1.0` | distance used for the standard `editconf -d` boxes (nm) |
| `--quantile` | `0.95` | target fraction of proteins whose box is large enough |
| `--candidates` | all 9 | descriptors allowed in the rule (see below) |
| `--max-features` | `2` | most descriptors in one rule. Use 1 if you have < 40 proteins |
| `--test-fraction` | `0.25` | share of proteins held out |
| `--seed` | `0` | random split; change it to see how stable the result is |
| `--cv-repeats` | `20` | CV repetitions (lower = faster) |
| `--min-gain` | `0.02` | a bigger rule must save > 2 % volume to be preferred |
| `--tag` | settings | name of the output folder |

Descriptors (`--candidates`): `log_n_heavy`, `log_rg`, `asphericity`,
`log_axis_ratio_1_3`, `terminal_coil_max`, `coil_fraction`,
`log_nma_msf_10`, `log_inv_lambda7`, `mode7_localisation`.

### Console output

```text
[data] 60 proteins usable
[split] train=45  test=15
  cov=0.972  vol/cubic=0.591  (intercept only)
  cov=0.978  vol/cubic=0.521  log_n_heavy
  ...
[select] log_n_heavy+terminal_coil_max  (CV coverage 0.972, median vol/cubic 0.490)

Held-out test proteins
method            coverage  median vol/cubic  worst short (nm)
fixed_A            100.00%             0.620            -0.130
model               93.33%             0.471             0.082
standard_cubic     100.00%             1.000            -0.208
standard_rect      100.00%             0.681            -0.178
```

*(Example from synthetic data. Only the format carries over.)*

How to read the final table:

* **coverage**: share of test proteins whose box was large enough on all
  three axes, i.e. no periodic-image contact closer than `--cutoff` in any
  frame of any replica.
* **median vol/cubic**: box volume relative to the standard cubic box
  (`editconf -bt cubic -d 1.0`). 0.47 = 53 % less volume, roughly that much
  less water to simulate.
* **worst short**: the largest amount by which any test box was too short
  (nm). Negative = every box had room to spare.
* Methods: `model` = your rule. `fixed_A` = one amplitude for everyone
  (option 1 from our discussion). `standard_cubic`, `standard_rect` = the
  usual `editconf -d` boxes.

The result you want: `model` with coverage close to `standard_cubic` and a
clearly smaller volume, and better than `fixed_A`.

### Output folder: `results/calibration/<tag>/`

Default tag: `<frame>_<stat>_<scaling>_K<n>_q<quantile>`, e.g.
`lab_max_thermal_K10_q0.95`.

| File | Content |
|---|---|
| `summary.json` | everything needed for the paper: settings, `n_proteins/n_train/n_test`, `selected_features`, `fixed_A_nm`, `cv_of_selected`, and `test` = per method `coverage`, `median_/mean_/p90_volume_vs_cubic`, `worst_shortfall_nm` |
| `model.json` | the fitted rule: `features`, `standardisation_mean/sd`, `coef_standardised`, `intercept`, `conformal_shift`, and the formula in `predict` |
| `dataset.csv` | one row per protein: `A_star`, `A_x/y/z`, `req_x/y/z` (room needed), `extrapolated` (A\* beyond the grid), all descriptors |
| `feature_search_cv.csv` | every descriptor combination tried, with its CV `coverage`, volumes, `raw_coverage_before_shift`, `conformal_shift` |
| `test_predictions.csv` | test proteins × methods: `A_pred`, `A_star`, box `Lx/Ly/Lz`, `safe`, `worst_shortfall_nm`, `volume_nm3`, `volume_vs_cubic` |
| `A_star_vs_features.png` | A\* against the chosen descriptors |
| `test_predicted_vs_needed.png` | predicted vs needed amplitude on test proteins; red = box too small |
| `test_methods_volume.png` | volume relative to cubic per method, with coverage under each label |

The rule in `model.json`, written out:

```text
A = max( exp( intercept + conformal_shift
              + Σ_j coef_j · (x_j − mean_j) / sd_j ) − 0.01 , 0 )      [nm]
box side L_a = NM envelope size along a at amplitude A  +  cutoff
```

---

## 4. Choosing settings without fooling yourself

You will want to compare `lab` vs `body`, `thermal` vs `uniform`, and K = 5,
10 or 20. Each run makes its own folder, so just run them:

```bash
for f in lab body; do for s in thermal uniform; do for k in 5 10 20; do
  python calib_fit_model.py --pdb-list passing_pdbs.txt \
      --frame $f --scaling $s --n-modes $k --cv-repeats 10
done; done; done
```

**Choose between settings using the CV numbers** (`cv_of_selected` in
`summary.json`), **not the test numbers.** If you pick the setting with the
best test result, the test set is no longer independent and the reported
coverage will be too optimistic. Decide on the setting from CV, then report
that setting's test result. Keep `--seed` fixed while comparing, then rerun
the chosen setting with a few seeds to show the split doesn't drive the result.

Which frame to report: `lab` answers "what box does a normal MD simulation
need" and is the honest default. `body` gives smaller boxes but is only safe
if the protein's rotation is restrained during MD.

---

## 5. Troubleshooting

| Message | Cause / fix |
|---|---|
| `No box in ... cannot unwrap PBC` | the XTC has no box information; such trajectories can't be used |
| `raw_modes_all has N rows; reference has M atoms` | modes computed on a different atom set; rerun with `--run-nma` |
| `[warn] ... needed extrapolation beyond the amplitude grid` | rerun step 2 for those proteins with a larger `--amp-max` |
| `[warn] dropping ... envelope never reaches the requirement` | the NM box can't cover that protein at any amplitude (e.g. it unfolds or a large domain swings far). Worth looking at individually and mentioning in the paper |
| `[warn] feature X has missing values; not used` | a descriptor failed for some protein, often DSSP for unusual residues; check `features.json` |
| `[warn] fewer than 20 proteins` | results will be very noisy; use `--max-features 1` |

---

## 6. Not implemented yet

* **Predicting a box for a new protein without MD.** Step 2 currently takes
  its reference structure from step 1, which needs MD. A small
  `calib_predict.py` (structure + NMA → descriptors → amplitude from
  `model.json` → box) is the missing piece for the "use it" part of the
  paper.
* Tumbling is not modelled; with `--frame lab` the calibration holds for
  simulations of similar length to the training data.
* Rectangular boxes only (no dodecahedron).
* Bio3D `U` is used as the modes; check whether mass-weighting means
  `modes` should be used instead.
