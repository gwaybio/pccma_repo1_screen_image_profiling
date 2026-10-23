# %% [markdown]
# # Aggregate bulk profiles, {SCREEN}
#
# Aggregates each plate's annotated cells to one profile per well (the median of the cells) and writes it to
# `data/bulk_profiles/<plate>_bulk_aggregated.parquet`. These per-plate, un-normalized profiles are the input of
# `3b.pooled_bulk_processing`, where the corrections, normalization and feature selection happen once over the whole screen.
# Platemap columns that are constant within a well are kept. The processing is `bulk_pipeline.aggregate_plates`.
#
# Set `PLATE_ID` to process one plate and `OVERWRITE=1` to redo plates that have output.

# %% [markdown]
# ## Run

# %%
import sys

sys.path.append("../../utils")   # bulk_pipeline.py, shared by every screen
sys.path.append(".")             # this screen's screen_config.py

import bulk_pipeline
from screen_config import config

# %%
bulk_pipeline.aggregate_plates(config)
