# %% [markdown]
# # Annotate profiles, {SCREEN}
#
# Attaches the platemap, drops the poor-quality segmentations (the per-cell flags from `1.single_cell_qc`) and writes one annotated
# profile per plate to `data/annotated_profiles/<plate>_annotated.parquet`. The cells are dropped before annotating, which keeps
# the memory use down. Everything that differs between screens (paths, platemap columns, join keys) is in `screen_config.py`;
# the processing is `bulk_pipeline.annotate_plates`.
#
# Set `PLATE_ID` to process one plate (run_pipeline.sh does, one process per plate) and `OVERWRITE=1` to redo plates that have output.

# %% [markdown]
# ## Run

# %%
import sys

sys.path.append("../../utils")   # bulk_pipeline.py, shared by every screen
sys.path.append(".")             # this screen's screen_config.py

import bulk_pipeline
from screen_config import config

# %%
bulk_pipeline.annotate_plates(config)
