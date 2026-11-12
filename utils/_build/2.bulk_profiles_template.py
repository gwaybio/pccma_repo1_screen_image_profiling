# %% [markdown]
# # Bulk profiles
#
# From each plate's quality-controlled single cells to the screen's bulk profiles. This notebook is the same in every screen:
# what differs between screens (paths, platemap columns, the plate layout, thresholds) is in `screen_config.yaml` next to it,
# and the code is in `utils/bulk_pipeline.py` (the steps) and `utils/bulk_utils.py` (the corrections). Run it top to bottom.
#
# | Step | What | Per | Writes |
# |---|---|---|---|
# | `annotate` | attach the platemap; drop the poor-quality segmentations (the flags of `1.single_cell_qc`) | plate | `data/annotated_profiles/<plate>_annotated.parquet` |
# | `aggregate` | the median of each feature over the cells of every well; platemap columns constant within a well are kept | plate | `data/bulk_profiles/<plate>_bulk_aggregated.parquet` |
# | `counts` | quality-controlled cells per well, the covariate of the cell-count effect | screen | `<screen>_well_cell_counts.csv` on the mount |
# | `pool` | corrections, normalization, rank transform and feature selection over all plates (table below) | screen | the bulk profiles and the correction tables on the mount |
#
# Environment variables: `PLATE_ID` limits `annotate` and `aggregate` to one plate (`run_pipeline.sh` runs one process per plate,
# which keeps memory down); `OVERWRITE=1` redoes finished plates; `STEPS` is a comma-separated subset of the four steps
# (default: all); `PROFILES_DIR`, `ANNOTATED_DIR` and `BULK_PROFILES_DIR` override where the whole-screen steps read and write;
# `ALLOW_PARTIAL=1` lets `pool` run without every plate. Existing per-plate outputs are skipped.

# %% [markdown]
# ## Configuration

# %%
import dataclasses
import os
import pprint
import sys

sys.path.append("../../utils")   # bulk_pipeline.py, shared by every screen

import bulk_pipeline

config = bulk_pipeline.load_config("screen_config.yaml")
pprint.pprint(dataclasses.asdict(config), sort_dicts=False)

# %%
ALL_STEPS = ["annotate", "aggregate", "counts", "pool"]
steps = [s for s in os.environ.get("STEPS", ",".join(ALL_STEPS)).split(",") if s]
assert set(steps) <= set(ALL_STEPS), f"STEPS must be a subset of {ALL_STEPS}, got {steps}"
print("running:", steps)

# %% [markdown]
# ## Per plate: annotate
#
# Attaches the platemap, drops the poor-quality segmentations (the per-cell flags from `1.single_cell_qc`) and writes one
# annotated profile per plate. The cells are dropped before annotating, which keeps the memory use down.

# %%
if "annotate" in steps:
    bulk_pipeline.annotate_plates(config)

# %% [markdown]
# ## Per plate: aggregate
#
# Aggregates each plate's annotated cells to one profile per well (the median of the cells). These per-plate, un-normalized
# profiles are the input of `pool`, where the corrections, normalization and feature selection happen once over the whole screen.

# %%
if "aggregate" in steps:
    bulk_pipeline.aggregate_plates(config)

# %% [markdown]
# ## Whole screen: cells per well
#
# Counts the quality-controlled single cells in every well. Only the plate and well columns of each annotated profile are
# read, so this is fast. Plates already in the output are skipped. The annotated profiles are read from
# `data/annotated_profiles` if it has any, else from the screen's folder on the bandicoot mount.

# %%
if "counts" in steps:
    bulk_pipeline.count_cells_per_well(config)

# %% [markdown]
# ## Whole screen: pool and process
#
# Turns the per-plate well profiles into the screen's bulk profiles, using the order *correct, then normalize, then
# transform, then select*. Two profile sets are written: the *corrected* set (steps 1-11; the main output) and a *base* set
# that skips steps 7-8. The shifts (steps 3-4) and effects (steps 7-8) are saved per well, and the plate statistics per plate,
# so that `4b.single_cell_correct` (next PR) applies exactly the same constants to every single cell. No plate is filtered out.
#
# | Step | What | Kind | Code |
# |---|---|---|---|
# | 1 | Pool the plates; drop empty wells (no compound, no solvent); read each plate's layout from the platemap files | | `bulk_utils.pool_plate_profiles`, `layouts_from_platemaps` |
# | 2 | Drop every feature that is missing or infinite in any well | | `bulk_utils.drop_nonfinite_features` |
# | 3 | Line up the two DMSO columns (2 and 24) of every plate with each other | correction (E3a) | `bulk_utils.align_control_columns` |
# | 4 | Match the DMSO columns to the interior compound wells, per plate layout | correction (E3) | `bulk_utils.match_outer_columns` |
# | 5 | Keep only features that vary in the DMSO wells of every plate | filter | `bulk_utils.variant_features` |
# | 6 | Provisional per-plate scaling on DMSO (used only to estimate step 7 on a common scale) | | `bulk_utils.plate_scale` |
# | 7 | Estimate the cell-count effect and one shared plate-position map with an amplitude per layout (as in the CFReT screen); DMSO wells get the row part of the map | correction | `bulk_utils.estimate_effects` |
# | 8 | Subtract those effects (in original units) | correction | |
# | 9 | Normalize: centre each plate on its corrected DMSO median, divide by 1.4826 x MAD | normalization | `bulk_utils.plate_scale`, `apply_plate_scale` |
# | 10 | Rank-based inverse normal transform over all wells (bulk only) | transform | `bulk_utils.inverse_normal_transform` |
# | 11 | Feature selection: near-constant, near-duplicate (correlation 0.9), blocklisted, missing | selection | `bulk_utils.select_features` |

# %%
if "pool" in steps:
    bulk_pipeline.run_pooled_bulk_processing(config)

# %% [markdown]
# ### Step 1: pool the plates
#
# One row per well, all plates stacked, with the values as aggregated (a per-well median of the quality-controlled single cells).
# Every plate listed in the screen's barcode-to-platemap file must be there, or the step stops; `ALLOW_PARTIAL=1` makes it warn
# and process the plates found (for a deliberate run on some plates).
#
# ### Step 3: line up the two DMSO columns of every plate (E3a)
#
# In these screens DMSO sits in columns 2 and 24 of every plate (`control_columns`). Both hold the same substance, so within a plate they should be
# interchangeable, but each plate's column 2 differs from its column 24 by a stable, plate-specific offset (it holds
# across rows and changes sign between plates, so no single correction for the screen can remove it). Within each plate
# and for each feature, the DMSO wells of both columns are reshaped, keeping their order, to follow the plate's pooled
# DMSO distribution. Only DMSO wells are touched.
#
# ### Step 4: match the DMSO columns to the interior compound wells (E3)
#
# For each layout and each feature, the DMSO wells in the outer columns (`outer_columns`; 2 and 24 here, the empty
# columns 1 and 23 are not DMSO) are reshaped, keeping their order, to the distribution of that layout's compound wells in
# the interior columns (`interior_columns`; 3-22 here).
# This is done on the original, unscaled values. It makes the outer-column DMSO wells representative of the interior.
#
# ### Step 5: variant filter
#
# A feature is kept only if, on **every** plate, the DMSO wells have a MAD that is not 0 and |MAD / median| above
# `min_abs_coef_var`; otherwise the normalization in step 9 would divide by about zero.
#
# ### Step 6: provisional scaling (to estimate the effects on a common scale)
#
# Each plate is centred on its DMSO median and divided by 1.4826 x its DMSO MAD, so that plates share units while the
# cell-count and position effects are estimated in step 7. These values are *not* the final normalization (step 9).
#
# ### Step 7: estimate the cell-count effect and the position map
#
# * **Cell count.** In DMSO wells, the part of each feature that moves with the number of cells in the well, within a
#   plate and column (DMSO wells are identical, so this is artifact; it produces the alternating-row pattern in column 2).
# * **Position.** One *shared* additive row effect and column effect per feature, from the replicate-averaged compound wells
#   of all layouts after the cell-count effect (medians, because values are heavy-tailed). DMSO wells (the control columns,
#   where no compound sits) receive the row part of the map; their column level was set by step 4.
# * **Amplitude per layout.** The position effect is not equally strong on every layout, so each layout gets the scale at
#   which the shared map best fits its own compound wells (least squares), with a bootstrap standard error, shrunk toward
#   the other layouts' amplitudes. A well's position effect is `amplitude x map(row, column)`.
#
# ### Step 8: subtract the effects, in original units
#
# The effects were estimated in the provisional units of step 6, so they are multiplied back by each plate's scale and
# subtracted from the matched values. This is a constant per well and feature, so it can be applied to every cell.
#
# ### Step 9: normalize each plate on its DMSO wells
#
# The plate's DMSO median and 1.4826 x MAD are taken from the *corrected* DMSO wells (for the base set, from the matched
# ones), and every well is centred and scaled with them. The same constants are applied to single cells in
# `4b.single_cell_correct` (next PR), so a well's bulk profile equals the median of its normalized cells.
#
# ### Step 10: inverse normal transform (bulk profiles only)
#
# pycytominer's `normalize(method="inverse_normal")` over all wells of the screen: each feature is rank-transformed to a
# standard normal (interpolating between quantile landmarks). Single cells do not get this step.
#
# ### Step 11: feature selection
#
# pycytominer's `feature_select`, on all wells together: frequency filter, correlation filter, blocklist, drop features
# with missing values.
#
# ### Save the tables that single cells need
#
# Per well (plate, well): the shift from steps 3-4 and the effects from step 8, both added/subtracted in original
# units. Per plate: the centre and scale from step 9 (and the base set's). Plus the feature lists.
