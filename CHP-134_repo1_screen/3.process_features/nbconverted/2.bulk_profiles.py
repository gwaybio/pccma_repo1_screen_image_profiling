import dataclasses
import os
import pprint
import sys

sys.path.append("../../utils")   # bulk_pipeline.py, shared by every screen

import bulk_pipeline

config = bulk_pipeline.load_config("screen_config.yaml")
pprint.pprint(dataclasses.asdict(config), sort_dicts=False)

ALL_STEPS = ["annotate", "aggregate", "counts", "pool"]
steps = [s for s in os.environ.get("STEPS", ",".join(ALL_STEPS)).split(",") if s]
assert set(steps) <= set(ALL_STEPS), f"STEPS must be a subset of {ALL_STEPS}, got {steps}"
print("running:", steps)

if "annotate" in steps:
    bulk_pipeline.annotate_plates(config)

if "aggregate" in steps:
    bulk_pipeline.aggregate_plates(config)

if "counts" in steps:
    bulk_pipeline.count_cells_per_well(config)

if "pool" in steps:
    bulk_pipeline.run_pooled_bulk_processing(config)
