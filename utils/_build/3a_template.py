# %% [markdown]
# # Cells per well, {SCREEN}
#
# Counts the quality-controlled single cells in every well. The count is the covariate of the cell-count effect that
# `3b.pooled_bulk_processing` estimates and removes. Only the plate and well columns of each annotated profile
# are read, so this is fast. Plates already in the output are skipped (set `OVERWRITE=1` to recount everything).
# The annotated profiles are read from `data/annotated_profiles` (the output of `2.annotate`) if it has any, else from the
# screen's folder on the bandicoot mount; `ANNOTATED_DIR` overrides this. The counts are written to the mount (`PROFILES_DIR`
# overrides it). The processing is `bulk_pipeline.count_cells_per_well`.

# %% [markdown]
# ## Run

# %%
import sys

sys.path.append("../../utils")   # bulk_pipeline.py, shared by every screen
sys.path.append(".")             # this screen's screen_config.py

import bulk_pipeline
from screen_config import config

# %%
bulk_pipeline.count_cells_per_well(config)
