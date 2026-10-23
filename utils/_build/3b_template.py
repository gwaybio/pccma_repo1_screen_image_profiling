# %% [markdown]
# # Pooled bulk processing, {SCREEN}
#
# Turns the per-plate well profiles (the median of every feature over a well's quality-controlled cells) into the
# screen's bulk profiles, using the order *correct, then normalize, then transform, then select*:
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
#
# Two profile sets are written: the *corrected* set (steps 1-11; the main output) and a *base* set that skips steps 7-8.
# The shifts (steps 3-4) and effects (steps 7-8) are saved per well, and the plate statistics per plate, so that
# `4b.single_cell_correct` applies exactly the same constants to every single cell. No plate is filtered out.
#
# The processing is `bulk_pipeline.run_pooled_bulk_processing` (using `bulk_utils`); the steps are explained below.
# `PROFILES_DIR` overrides where the files are read and written and `BULK_PROFILES_DIR` where the per-plate aggregates are read from.

# %% [markdown]
# ## Run

# %%
import sys

sys.path.append("../../utils")   # bulk_pipeline.py, shared by every screen
sys.path.append(".")             # this screen's screen_config.py

import bulk_pipeline
from screen_config import config

# %%
bulk_pipeline.run_pooled_bulk_processing(config)

# %% [markdown]
# ### Step 1: pool the plates
#
# One row per well, all plates stacked, with the values as aggregated (a per-well median of the quality-controlled single cells).
#
# ### Step 3: line up the two DMSO columns of every plate (E3a)
#
# DMSO sits in columns 2 and 24 of every plate. Both hold the same substance, so within a plate they should be
# interchangeable, but each plate's column 2 differs from its column 24 by a stable, plate-specific offset (it holds
# across rows and changes sign between plates, so no single correction for the screen can remove it). Within each plate
# and for each feature, the DMSO wells of both columns are reshaped, keeping their order, to follow the plate's pooled
# DMSO distribution. Only DMSO wells are touched.
#
# ### Step 4: match the DMSO columns to the interior compound wells (E3)
#
# For each layout and each feature, the DMSO wells in the outer columns
# (2 and 24) are reshaped, keeping their order, to the distribution of that layout's compound wells in columns 3-22.
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
#   of all layouts after the cell-count effect (medians, because values are heavy-tailed). DMSO wells (columns 2 and 24,
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
# `4b.single_cell_correct`, so a well's bulk profile equals the median of its normalized cells.
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
