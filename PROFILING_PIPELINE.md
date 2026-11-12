# Image-based profiling pipeline (CHP-134 and SK-N-AS), as implemented

Starts from single cells that passed cell-level QC (coSMicQC). No plate filtering. Written from the code in
`utils/profiling/bulk_pipeline.py`, `utils/profiling/bulk_utils.py`, `utils/profiling/single_cell_utils.py` and the notebooks `2.bulk_profiles_<cell line>`, `4b` and `5b`; each
has a `nbconverted/*.py` twin that `run_pipeline.sh` calls (made by `jupyter nbconvert`). The bulk steps (numbered
2, 3, 3a and 3b below) are the steps `annotate`, `aggregate`, `counts`, and `pool`, `correct`, `normalize` of the one notebook
`2.bulk_profiles_<cell line>`, which is the same for every screen except for the cell line (its title and the configuration file it loads): the code is in `bulk_pipeline.py` and
what differs between screens is in each screen's `screen_config_<cell line>.yaml` (next section).

## Configuration: `screen_config_<cell line>.yaml`

One file per screen, in its `3.process_features` folder, holds what differs between screens; `bulk_pipeline.load_config` reads it
and rejects missing or unknown keys. It has three groups of keys:

| Group | Keys |
|---|---|
| the screen | `cell_line`, `screen_name` (prefixes every saved table), `profiles_dir` (the screen's folder on the mount; `PROFILES_DIR` overrides it) |
| inputs | `raw_profile_dirs` (candidate folders; the first that exists is used), `raw_profile_suffix`, `platemap_dir`, `barcode_platemap_file`, `platemap_columns`, `platemap_rename`, `annotate_join_on`, `nuclei_location_columns` |
| per-plate outputs | `qc_dir`, `annotated_dir`, `bulk_dir`, `annotated_subdir`, `annotated_glob`, `per_plate_glob` |

A new screen needs a new `screen_config_<cell line>.yaml` and a copy of the notebook for its cell line, not new code. (In file names the cell line has no dash: `CHP134`, `SKNAS`.)

**The plate layout is not configured; it is read from the platemaps** (`bulk_utils.design_from_platemaps`). A well with a batch ID
is a compound well; a well with no batch ID but a solvent is a control well; a well with neither is empty. The interior columns are the
columns that have compound wells, the control columns are the ones that have control wells, every other column is an outer column,
and the number of rows comes from the well names. The step stops if the platemaps disagree about the control columns, if a control
column also holds compounds, or if the compound columns are not consecutive (the column-wise corrections below need controls kept
apart from compounds). For both screens this gives 16 rows, control columns 2 and 24, interior columns 3-22, outer columns 1, 2, 23, 24.

**There are no thresholds in the file.** The variant filter (`min_abs_coef_var` 1e-3), the correlation filter (0.9), the cell-count
winsorizing, the bootstrap and backfitting counts and the frequency-filter cut-offs are parameters of the functions in `bulk_utils.py`,
with the values listed below; they are the same for every screen. A group of wells too small to define a distribution (fewer than
`MIN_WELLS` = 2) is an error, not a silent skip.

## Facts about the plates (after empty wells are dropped)

Each plate has DMSO-only wells in columns 2 and 24 and compound-only wells in columns 3-22. Columns 1 and 23 are empty.
A *layout* is a set of plates with the same compound in the same wells (CHP-134: 9, SK-N-AS: 11; SK-N-AS has one platemap
file per plate, so layouts are read from the files' contents). Layouts come from `0.download_data/metadata/barcode_platemap.csv`.

## Environment

uv, Python 3.12, `pyproject.toml` / `uv.lock` (pycytominer 1.7.1, pandas 2.3.3, numpy 2.3.5, scipy 1.18.1, scikit-learn 1.7.2,
pyarrow 20). Create it with `uv sync`; `run_pipeline.sh` and `utils/run_full_profiling.sh` use `.venv`. Checks against
pycytominer: `uv run pytest` (the tests are in `tests/`).

## Where things are written (bandicoot; existing files are never overwritten)

| Folder | New files |
|---|---|
| `<screen profiles>/bulk_profiles/` (CHP-134: `CHP-134_repo1_profiles/profiles/`; SK-N-AS: `SK-N-AS_repo1_profiles/`) | `<screen>_well_cell_counts.csv`; `<screen>_pooled_bulk_corrected_feature_selected.parquet` (main); `<screen>_pooled_bulk_base_feature_selected.parquet`; `<screen>_correction_well_shift.parquet`, `_correction_well_effects.parquet`, `_correction_plate_scale_corrected.parquet`, `_correction_plate_scale_base.parquet`, `_correction_features.json`, `_correction_layout_amplitudes.csv`; `<screen>_sc_selected_features_corrected.json` |
| `<screen profiles>/single_cell_profiles/` | `<plate>_sc_corrected_stats.pkl` (intermediate); `<plate>_sc_corrected_feature_selected.parquet` (final single cells); timing logs |
| `<screen profiles>/run_logs/` | status file and one log per step and plate |

Large files are written to local scratch, copied, and accepted only if a checksum read back from the mount matches
(`utils/profiling/io_utils.py`): a large direct write to the mount was once seen to lose 16 MiB regions silently.

## Run order (`utils/run_full_profiling.sh`)

| Order | Step | Per | Reads | Writes |
|---|---|---|---|---|
| 1 | `1.single_cell_qc` | plate | cells | QC flags |
| 2 | `2.bulk_profiles_<cell line>` `STEPS=annotate` | plate | profiles, QC flags, platemap | `<plate>_annotated.parquet` (poor-quality cells dropped) |
| 3 | `2.bulk_profiles_<cell line>` `STEPS=aggregate` | plate | annotated cells | `<plate>_bulk_aggregated.parquet` |
| 3a | `2.bulk_profiles_<cell line>` `STEPS=counts` | screen | annotated cells (plate, well columns) | counts csv |
| 3b | `2.bulk_profiles_<cell line>` `STEPS=pool,correct,normalize` | screen | per-plate aggregates, counts, platemap files | bulk profiles (corrected, base), correction tables |
| 4b-stats | `4b.single_cell_correct` `STAGE=stats` | plate | annotated cells (all variant features), tables | statistics pickle (no cells) |
| 5b | `5b.single_cell_feature_select` | screen | all plates' statistics | selected-features JSON |
| 4b-apply | `4b.single_cell_correct` `STAGE=apply` | plate | annotated cells (selected features), tables, list | corrected single cells |

The per-plate single-cell steps run two plates at a time (`WORKERS`). Single cells are never rank-transformed and never
sphered. pycytominer is used for plate scaling (`RobustMAD`), the inverse normal transform and every feature-selection
filter; the corrections and the control-based variant filter have no pycytominer equivalent and are in `utils/profiling/bulk_utils.py`.

## 3b: bulk profiles and tables

The notebook runs this as three steps that hand the data to each other in memory: `pool` (steps 1-2), `correct` (steps 3-8; writes the
shift, effect and amplitude tables) and `normalize` (steps 9-11; writes the bulk profiles, plate scales and feature lists).

| # | Step (function) | Parameters | Hands on |
|---|---|---|---|
| 1 | Pool; drop empty wells; read layouts (`pool_plate_profiles`, `drop_wells`, `layouts_from_platemaps`) | empty = no compound and no solvent | wells x features; layout per plate |
| 2 | Drop non-finite features | NaN/inf in any well | `original` values kept |
| 3 | E3a align DMSO columns (`align_control_columns`) | the control columns (2, 24); at least 2 controls per column per plate; rank-preserving map onto the plate's pooled controls | aligned values |
| 4 | E3 match to interior (`match_outer_columns`) | per layout; baseline = layout's compound wells in the interior columns (3-22); outer columns 1, 2, 23, 24 (columns with no wells are skipped) | matched values; `shift = matched - original` |
| 5 | Variant filter (`plate_control_stats`, `variant_features`) | control MAD not 0 and abs(MAD/median) > 1e-3 on every plate | variant features |
| 6 | Provisional scale (`plate_scale`, `apply_plate_scale`) | per plate: (x - control median) / (1.4826 x control MAD) | values for step 7 only |
| 7 | Effects (`estimate_effects`) | below | `density`, `position`, amplitudes |
| 8 | Subtract effects, original units | `effects = (density + position) x provisional plate scale` | corrected values |
| 9 | Normalize | per plate: (x - median) / (1.4826 x MAD) of the **corrected** controls | normalized wells; final plate scale |
| 10 | Inverse normal transform (`inverse_normal_transform`) | pycytominer `inverse_normal`, all wells, seed 0 | transformed wells |
| 11 | Feature selection (`select_features`) | frequency (freq_cut 0.05, unique_cut 0.01), correlation 0.9, blocklist, drop NaN | bulk profiles |

The base set skips steps 7-8.

3b stops if a plate listed in the screen's `barcode_platemap.csv` has no per-plate file, and `run_pipeline.sh` skips 3b when a
plate's aggregation or the cell counts failed. `ALLOW_PARTIAL=1` overrides both, for a deliberate run on some plates (3b then warns).

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

- `tests/test_bulk_utils.py`: the plate scaling equals pycytominer's `mad_robustize` (to 1e-12); the plate design is read correctly from both screens' platemaps and from a synthetic 96-well design; platemaps that cannot be described, a missing plate and a group of fewer than two wells are errors.
- `tests/test_bulk_pipeline.py`: the screen configurations load and bad ones are rejected; `correct` and `normalize` say which step is missing.
- `tests/test_single_cell_utils.py`: the streaming feature selection equals `pycytominer.feature_select` on the pooled cells.
- 2026-10-23: CHP-134 annotate, aggregate, 3a and 3b rerun through `bulk_pipeline.py` reproduce the existing per-plate
  profiles, the corrected and base bulk profiles (9,400 wells) and the shift and effect tables exactly (maximum difference 0).
- 2026-10-21: every stage of `utils/run_full_profiling.sh` finished for both screens with no missing or unreadable file
  (`utils/verify_outputs.py`); per-well medians of the corrected single cells equal the bulk profile by Spearman 1.000 on one
  plate per screen (80 features).
