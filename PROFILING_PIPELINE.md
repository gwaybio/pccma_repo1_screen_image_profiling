# Image-based profiling pipeline (CHP-134 and SK-N-AS), as implemented

Starts from single cells that passed cell-level QC (coSMicQC). No plate filtering. Written from the code in
`utils/bulk_pipeline.py`, `utils/bulk_utils.py`, `utils/single_cell_utils.py` and the notebooks `2`, `3`, `3a`, `3b`, `4b`, `5b`; each
has a `nbconverted/*.py` twin that `run_pipeline.sh` calls (generated from `utils/_build/*_template.py`). Steps 2-3b are the
same for every screen: the code is in `bulk_pipeline.py` and what differs between screens is in each screen's `screen_config.py`.

Rule: **correct, then normalize, then (bulk only) rank-transform, then select features.** Corrections remove one named
artifact each and are constants per well, so they apply identically to a well's median and to every cell in it.

## Facts about the plates (after empty wells are dropped)

Each plate has DMSO-only wells in columns 2 and 24 and compound-only wells in columns 3-22. Columns 1 and 23 are empty.
A *layout* is a set of plates with the same compound in the same wells (CHP-134: 9, SK-N-AS: 11; SK-N-AS has one platemap
file per plate, so layouts are read from the files' contents). Layouts come from `0.download_data/metadata/barcode_platemap.csv`.

## Environment

uv, Python 3.12, `pyproject.toml` / `uv.lock` (pycytominer 1.7.1, pandas 2.3.3, numpy 2.3.5, scipy 1.18.1, scikit-learn 1.7.2,
pyarrow 20). Create it with `uv sync`; `run_pipeline.sh` and `utils/run_full_profiling.sh` use `.venv`. Checks against
pycytominer: `uv run python utils/test_profiling_utils.py` (bulk) and `uv run python utils/test_single_cell_utils.py` (single cells).

## Where things are written (bandicoot; existing files are never overwritten)

| Folder | New files |
|---|---|
| `<screen profiles>/bulk_profiles/` (CHP-134: `CHP-134_repo1_profiles/profiles/`; SK-N-AS: `SK-N-AS_repo1_profiles/`) | `<screen>_well_cell_counts.csv`; `<screen>_pooled_bulk_corrected_feature_selected.parquet` (main); `<screen>_pooled_bulk_base_feature_selected.parquet`; `<screen>_correction_well_shift.parquet`, `_correction_well_effects.parquet`, `_correction_plate_scale_corrected.parquet`, `_correction_plate_scale_base.parquet`, `_correction_features.json`, `_correction_layout_amplitudes.csv`; `<screen>_sc_selected_features_corrected.json` |
| `<screen profiles>/single_cell_profiles/` | `<plate>_sc_corrected_stats.pkl` (intermediate); `<plate>_sc_corrected_feature_selected.parquet` (final single cells); timing logs |
| `<screen profiles>/run_logs/` | status file and one log per step and plate |

Large files are written to local scratch, copied, and accepted only if a checksum read back from the mount matches
(`utils/io_utils.py`): a large direct write to the mount was once seen to lose 16 MiB regions silently.

## Run order (`utils/run_full_profiling.sh`)

| Order | Step | Per | Reads | Writes |
|---|---|---|---|---|
| 1 | `1.single_cell_qc` | plate | cells | QC flags |
| 2 | `2.annotate` | plate | profiles, QC flags, platemap | `<plate>_annotated.parquet` (poor-quality cells dropped) |
| 3 | `3.aggregate_bulk_profiles` | plate | annotated cells | `<plate>_bulk_aggregated.parquet` |
| 3a | `3a.well_cell_counts` | screen | annotated cells (plate, well columns) | counts csv |
| 3b | `3b.pooled_bulk_processing` | screen | per-plate aggregates, counts, platemap files | bulk profiles (corrected, base), correction tables |
| 4b-stats | `4b.single_cell_correct` `STAGE=stats` | plate | annotated cells (all variant features), tables | statistics pickle (no cells) |
| 5b | `5b.single_cell_feature_select` | screen | all plates' statistics | selected-features JSON |
| 4b-apply | `4b.single_cell_correct` `STAGE=apply` | plate | annotated cells (selected features), tables, list | corrected single cells |

The per-plate single-cell steps run two plates at a time (`WORKERS`). Single cells are never rank-transformed and never
sphered. pycytominer is used for plate scaling (`RobustMAD`), the inverse normal transform and every feature-selection
filter; the corrections and the control-based variant filter have no pycytominer equivalent and are in `utils/bulk_utils.py`.

## 3b: bulk profiles and tables

| # | Step (function) | Parameters | Hands on |
|---|---|---|---|
| 1 | Pool; drop empty wells; read layouts (`pool_plate_profiles`, `drop_wells`, `layouts_from_platemaps`) | empty = no compound and no solvent | wells x features; layout per plate |
| 2 | Drop non-finite features | NaN/inf in any well | `original` values kept |
| 3 | E3a align DMSO columns (`align_control_columns`) | columns 2, 24; at least 5 controls per column per plate; rank-preserving map onto the plate's pooled controls | aligned values |
| 4 | E3 match to interior (`match_outer_columns`) | per layout; baseline = layout's compound wells in columns 3-22; outer columns 1, 2, 23, 24; at least 20 interior, 5 outer wells | matched values; `shift = matched - original` |
| 5 | Variant filter (`plate_control_stats`, `variant_features`) | control MAD not 0 and abs(MAD/median) > 1e-3 on every plate | variant features |
| 6 | Provisional scale (`plate_scale`, `apply_plate_scale`) | per plate: (x - control median) / (1.4826 x control MAD) | values for step 7 only |
| 7 | Effects (`estimate_effects`) | below | `density`, `position`, amplitudes |
| 8 | Subtract effects, original units | `effects = (density + position) x provisional plate scale` | corrected values |
| 9 | Normalize | per plate: (x - median) / (1.4826 x MAD) of the **corrected** controls | normalized wells; final plate scale |
| 10 | Inverse normal transform (`inverse_normal_transform`) | pycytominer `inverse_normal`, all wells, seed 0 | transformed wells |
| 11 | Feature selection (`select_features`) | frequency (freq_cut 0.05, unique_cut 0.01), correlation 0.9, blocklist, drop NaN | bulk profiles |

The base set skips steps 7-8.

### Step 7 (CFReT-style: shared models, one amplitude per layout)

| Piece | Definition |
|---|---|
| Cell-count covariate | log cells per well from the counts file; wells without a count get no density effect |
| Compound wells | columns 3-22; covariate centred on the plate's compound mean, clipped to the plate's 5th-95th percentile |
| Control wells | columns 2 and 24; covariate centred within plate x column, clipped to the plate's 5th-95th percentile |
| Density slope | one per feature, from all control wells' within-(plate, column) variation |
| Position map | one per feature, from all compound wells after the density term and the plate mean; averaged per (compound, row, column); row (16) and column (20) levels by backfitting, each level the **median**, 6 iterations |
| Amplitude per layout | per feature, the least-squares slope of the layout's compound profiles on the map; the amplitude is the median of those slopes weighted by each feature's map energy; 200 bootstrap resamples for the standard error; shrunk toward the other layouts (empirical Bayes, as in CFReT) |
| Position effect | compound wells: `amplitude x map(row, column)`; DMSO wells in columns 2/24: `amplitude x row level` only |
| Exclusions | none |

## 4b and 5b: single cells

Per cell in well w of plate p: `x' = x + shift[p, w] - effects[p, w]`; `z = RobustMAD.transform(x')` with the plate's stored
median and scale (pycytominer's definition: `(x' - median) / (scale + 1e-18)`, scale = 1.4826 x MAD, the same constants as the
bulk profiles). The median of a well's cells therefore equals its bulk profile before the inverse normal transform.

Feature selection uses every cell without pooling them:

| Stage | What happens | Output |
|---|---|---|
| `stats` (per plate) | Correct and normalize all cells for every variant feature (blocks of 200 features; the plate's matrix is held in a temporary memory map in `SCRATCH_DIR`), then accumulate exact statistics: cell count, per-feature sum and NaN count, the cross-product matrix, value counts of discrete-looking features | one `.pkl` per plate |
| `5b` (screen) | Add the plates' statistics (exact), then run in order: `drop_na_columns` (any NaN), default `blocklist`, `frequency_threshold` (freq_cut 0.05, unique_cut 0.01), `variance_threshold` (1e-6), `correlation_threshold` (0.9) using pycytominer's own functions; the correlation filter runs on a small stand-in data frame with exactly the screen's correlation matrix | selected-features JSON |
| `apply` (per plate) | Read only the selected features, apply the same arithmetic, write float32 (metadata columns kept) | one parquet per plate |

Validated: on a synthetic matrix with missing values, dominant values, discrete features, near-constant features and correlated
duplicates, split over three "plates", this gives exactly the same features as `pycytominer.feature_select` on the pooled matrix.

## Checks

- `utils/test_profiling_utils.py`: the plate scaling equals pycytominer's `mad_robustize` (to 1e-12).
- `utils/test_single_cell_utils.py`: the streaming feature selection equals `pycytominer.feature_select` on the pooled cells.
- 2026-10-23: CHP-134 annotate, aggregate, 3a and 3b rerun through `bulk_pipeline.py` reproduce the existing per-plate
  profiles, the corrected and base bulk profiles (9,400 wells) and the shift and effect tables exactly (maximum difference 0).
- 2026-10-21: every stage of `utils/run_full_profiling.sh` finished for both screens with no missing or unreadable file
  (`utils/verify_outputs.py`); per-well medians of the corrected single cells equal the bulk profile by Spearman 1.000 on one
  plate per screen (80 features).

Not run on this version: the leave-one-layout-out check of the amplitude estimate.
