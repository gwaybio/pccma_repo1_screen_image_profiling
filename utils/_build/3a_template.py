# %% [markdown]
# # Cells per well, {SCREEN}
#
# Counts the quality-controlled single cells in every well. The count is the covariate of the cell-count effect that
# `3b.pooled_bulk_processing` estimates and removes. Only the plate and well columns of each annotated profile
# are read, so this is fast. Plates already in the output are skipped (set `OVERWRITE=1` to recount everything).
# `PROFILES_DIR` and `ANNOTATED_DIR` override where the files are read and written. The processing is
# `bulk_pipeline.count_cells_per_well`.

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
