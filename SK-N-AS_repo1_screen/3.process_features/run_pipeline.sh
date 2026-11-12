#!/bin/bash
#
# Run the full feature-processing pipeline, in order:
#   1.single_cell_qc.ipynb              -> one QC-annotations parquet per plate (via papermill)
#   2.bulk_profiles_SKNAS.ipynb STEPS=annotate  -> one annotated, QC-passed single-cell profile per plate
#   2.bulk_profiles_SKNAS.ipynb STEPS=aggregate -> per-plate aggregated (un-normalized) bulk profiles
#   2.bulk_profiles_SKNAS.ipynb STEPS=counts    -> cells per well (whole screen)
#   2.bulk_profiles_SKNAS.ipynb STEPS=pool,correct,normalize -> whole-screen bulk profiles and the well/plate correction tables
#                         (the notebook runs all six steps top to bottom; screen_config_SKNAS.yaml describes the screen)
#   4b.single_cell_correct.ipynb        -> STAGE=stats: statistics of every normalized single cell, per plate
#   5b.single_cell_feature_select.ipynb -> single-cell features selected on all cells' statistics (whole screen)
#   4b.single_cell_correct.ipynb        -> STAGE=apply: corrected, normalized single cells per plate
#
# Unlike CHP-134_repo1_screen's run_pipeline.sh, there is no merge_profiles
# step here: 0.convert_cytotable.ipynb (run separately, via
# convert_cytotable_hpc_parent.sh/convert_cytotable_hpc_child.sh) already
# writes one converted parquet per plate directly to the external drive,
# with no row-batch chunks to stitch together first.
#
# Annotation used to be duplicated -- once inside bulk processing, once inside
# single-cell processing -- so it ran (and held both a plate's raw profile and
# its annotated copy in memory at once, each tens of GB) twice per plate. It's
# now its own step (2.bulk_profiles_SKNAS.ipynb, STEPS=annotate) that both downstream branches read from,
# so that memory-heavy merge only happens once per plate.
#
# The per-plate steps all run one plate per `python` invocation (PLATE_ID env
# var) rather than looping over all plates inside a single long-lived process.
# Each plate's memory is fully released back to the OS when its process
# exits, which avoids the cross-plate memory buildup that can otherwise lead
# to an OOM kill partway through a batch. This matters even more here than
# for CHP-134, since each SK-N-AS plate (~55GB, 2.17M cells) is roughly 3x the
# size of a CHP-134 plate -- a single plate's raw profile alone has been
# measured at ~97GB in memory, so isolating each stage to its own process is
# load-bearing, not just a nice-to-have.
#
# Each plate's per-stage timing (and a total) is appended to a timing_log.csv
# under that step's output directory (data/annotated_profiles,
# data/bulk_profiles, or the single-cell stats / corrected folders).
#
# By default, every step skips a plate that already has output. Set
# OVERWRITE=1 (e.g. `OVERWRITE=1 ./run_pipeline.sh`) to reprocess and
# overwrite every plate's output instead.

set -uo pipefail

overwrite="${OVERWRITE:-}"

# Converted single-cell profiles live on the external drive, one parquet per
# plate, produced by 0.convert_cytotable.ipynb -- not under ./data like
# CHP-134's merged_profiles.
converted_dir="/media/18tbdrive2/SK-N-AS_repo1_profiles/converted_profiles"

# -----------------------------
# Total runtime tracking (printed on exit, success or failure, via trap)
# -----------------------------
pipeline_start_time=$(date +%s)

print_total_time() {
    local elapsed=$(( $(date +%s) - pipeline_start_time ))
    echo "======================================"
    printf 'Total pipeline time: %dh %dm %ds\n' \
        $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
    echo "======================================"
}
trap print_total_time EXIT

# -----------------------------
# Initialize environment
# -----------------------------
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ ! -x "$repo_root/.venv/bin/python" ]; then
    echo "uv environment not found at $repo_root/.venv; run 'uv sync' in $repo_root first; aborting."
    exit 1
fi
source "$repo_root/.venv/bin/activate"

failed_plates=()

# convert notebooks to scripts
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb
if [ $? -ne 0 ]; then
    echo "jupyter nbconvert FAILED; aborting pipeline before QC."
    exit 1
fi

mapfile -t plate_ids < <(find "$converted_dir" -maxdepth 1 -name "*_converted.parquet" -printf "%f\n" | sed 's/_converted\.parquet$//' | sort)

echo "Number of plates found: ${#plate_ids[@]}"

# -----------------------------
# Step 1: per-plate single-cell QC via papermill
# -----------------------------
echo "======================================"
echo "Step 1: 1.single_cell_qc.ipynb (papermill, per plate)"
echo "======================================"

qc_notebook_dir="./executed_notebooks/1.single_cell_qc"
mkdir -p "$qc_notebook_dir"

for plate_id in "${plate_ids[@]}"; do
    qc_output="./data/qc_results/${plate_id}_qc_annotations.parquet"
    executed_notebook="${qc_notebook_dir}/${plate_id}.ipynb"

    if [ -f "$qc_output" ] && [ -z "$overwrite" ]; then
        echo "✅ ${plate_id} already QC'd (found ${qc_output})"
        continue
    fi

    echo ">>> Running QC for ${plate_id}"
    # render_diagnostics=False skips the per-condition CytoDataFrame image
    # previews, which pull crops off the bandicoot network mount and are only
    # useful for interactive review -- the QC annotations export at the
    # bottom of the notebook doesn't depend on them. Leaving this on for a
    # batch run across many plates is what causes each plate to take a very
    # long time (or effectively hang) here.
    papermill 1.single_cell_qc.ipynb "$executed_notebook" \
        -p plate_id "$plate_id" \
        -p render_diagnostics False
    exit_code=$?

    if [ "$exit_code" -ne 0 ]; then
        echo "QC FAILED for plate: ${plate_id} (exit code $exit_code)"
        failed_plates+=("qc:${plate_id}")
    else
        echo "QC done for plate: ${plate_id}"
    fi
done

# -----------------------------
# Step 2: annotation (one plate/process at a time; skips plates without QC,
# and already-annotated plates). Shared input for both the bulk and
# single-cell branches below, so annotate() only runs once per plate.
# -----------------------------
echo "======================================"
echo "Step 2: 2.bulk_profiles_SKNAS.ipynb STEPS=annotate (one plate at a time)"
echo "======================================"

for plate_id in "${plate_ids[@]}"; do
    annotate_output="./data/annotated_profiles/${plate_id}_annotated.parquet"

    if [ -f "$annotate_output" ] && [ -z "$overwrite" ]; then
        echo "✅ ${plate_id} already annotated (found ${annotate_output})"
        continue
    fi

    echo ">>> Running annotation for ${plate_id}"
    STEPS=annotate PLATE_ID="$plate_id" OVERWRITE="$overwrite" python nbconverted/2.bulk_profiles_SKNAS.py
    exit_code=$?

    if [ "$exit_code" -ne 0 ]; then
        echo "Annotation FAILED for plate: ${plate_id} (exit code $exit_code)"
        failed_plates+=("annotate:${plate_id}")
    else
        echo "Annotation done for plate: ${plate_id}"
    fi
done

# -----------------------------
# Step 3: aggregation (one plate/process at a time; skips plates without
# an annotated profile, and already-processed plates)
# -----------------------------
echo "======================================"
echo "Step 3: 2.bulk_profiles_SKNAS.ipynb STEPS=aggregate (one plate at a time)"
echo "======================================"

for plate_id in "${plate_ids[@]}"; do
    bulk_output="./data/bulk_profiles/${plate_id}_bulk_aggregated.parquet"

    if [ -f "$bulk_output" ] && [ -z "$overwrite" ]; then
        echo "✅ ${plate_id} already aggregated (found ${bulk_output})"
        continue
    fi

    echo ">>> Running aggregation for ${plate_id}"
    STEPS=aggregate PLATE_ID="$plate_id" OVERWRITE="$overwrite" python nbconverted/2.bulk_profiles_SKNAS.py
    exit_code=$?

    if [ "$exit_code" -ne 0 ]; then
        echo "Aggregation FAILED for plate: ${plate_id} (exit code $exit_code)"
        failed_plates+=("aggregate:${plate_id}")
    else
        echo "Aggregation done for plate: ${plate_id}"
    fi
done

# -----------------------------
# Step 3a: cells per well (whole screen, run once). Counts the QC-passed cells in
# every well from the annotated profiles of step 2 (reads only the well column).
# 3b needs the counts to estimate the cell-count effect. Resumable: plates already in
# <screen profiles>/bulk_profiles/<screen>_well_cell_counts.csv (on the bandicoot mount) are skipped.
# -----------------------------
echo "======================================"
echo "Step 3a: 2.bulk_profiles_SKNAS.ipynb STEPS=counts (whole screen)"
echo "======================================"

echo ">>> Counting cells per well for the whole screen"
STEPS=counts OVERWRITE="$overwrite" python nbconverted/2.bulk_profiles_SKNAS.py
exit_code=$?

if [ "$exit_code" -ne 0 ]; then
    echo "Cell counting FAILED (exit code $exit_code)"
    failed_plates+=("well_cell_counts:whole_screen")
else
    echo "Cell counting done for whole screen"
fi

# -----------------------------
# Step 3b: pooled bulk processing (whole screen, run once). Pools every plate's
# un-normalized aggregated profile (from step 3) and processes it in the order correct,
# normalize, rank-transform, select (see 2.bulk_profiles_SKNAS.ipynb): drop NaN/inf
# features, line up and match the DMSO columns, per-plate variant filter, subtract the
# cell-count and plate-position effects, MAD normalization per plate, inverse normal
# transform, pooled feature selection. It reads the per-plate *_bulk_aggregated.parquet files
# of step 3 (from data/bulk_profiles, or the mount if that has none) and the cell counts from, and writes the <screen>_pooled_bulk_*_feature_selected.parquet
# profiles and the <screen>_correction_* tables to, <screen profiles>/bulk_profiles on the
# bandicoot mount (PROFILES_DIR overrides it). It always recomputes, overwriting its output.
# -----------------------------
echo "======================================"
echo "Step 3b: 2.bulk_profiles_SKNAS.ipynb STEPS=pool,correct,normalize (whole screen)"
echo "======================================"

# A failed aggregation or cell count would make 3b process a partial screen (or stale counts) without saying so, and
# overwrite the earlier whole-screen files with it. Skip 3b unless ALLOW_PARTIAL is set (a deliberate partial run).
blocking_failures=()
for failure in ${failed_plates[@]+"${failed_plates[@]}"}; do
    case "$failure" in
        aggregate:*|well_cell_counts:*) blocking_failures+=("$failure") ;;
    esac
done

if [ ${#blocking_failures[@]} -gt 0 ] && [ -z "${ALLOW_PARTIAL:-}" ]; then
    echo "Skipping pooled bulk processing: earlier steps failed for ${blocking_failures[*]}"
    echo "(fix them and rerun, or set ALLOW_PARTIAL=1 to process the plates that finished)"
    failed_plates+=("pooled_bulk_processing:skipped")
else
    echo ">>> Running pooled bulk processing for the whole screen"
    STEPS=pool,correct,normalize python nbconverted/2.bulk_profiles_SKNAS.py
    exit_code=$?

    if [ "$exit_code" -ne 0 ]; then
        echo "Pooled bulk processing FAILED (exit code $exit_code)"
        failed_plates+=("pooled_bulk_processing:whole_screen")
    else
        echo "Pooled bulk processing done for whole screen"
    fi
fi

# -----------------------------
# Step 4b-stats: single-cell statistics (one plate/process at a time)
# Applies the 3b constants to every QC-passed cell (all variant features) and accumulates exact statistics
# of the plate's cells for feature selection. Writes <screen profiles>/single_cell_profiles/<plate>_sc_<variant>_stats.pkl.
# Needs scratch space for the plate's normalized matrix (SCRATCH_DIR; about 21 GB for the largest SK-N-AS plate).
# -----------------------------
echo "======================================"
echo "Step 4b-stats: single-cell statistics (one plate/process at a time)"
echo "======================================"

variant="${VARIANT:-corrected}"
for plate_id in "${plate_ids[@]}"; do
    plate_key="${plate_id// /_}"
    echo ">>> Single-cell statistics for ${plate_id}"
    STAGE="stats" PLATE_ID="$plate_key" OVERWRITE="$overwrite" VARIANT="$variant" python nbconverted/4b.single_cell_correct.py
    exit_code=$?

    if [ "$exit_code" -ne 0 ]; then
        echo "Single-cell stats FAILED for plate: ${plate_id} (exit code $exit_code)"
        failed_plates+=("single_cell_stats:${plate_id}")
    else
        echo "Single-cell stats done for plate: ${plate_id}"
    fi
done

# -----------------------------
# Step 5b: single-cell feature selection (whole screen, run once). Adds up the plates'
# statistics and runs pycytominer's filters on all cells, independently of the bulk profiles; writes
# <screen profiles>/bulk_profiles/<screen>_sc_selected_features_<variant>.json.
# -----------------------------
echo "======================================"
echo "Step 5b: 5b.single_cell_feature_select.ipynb (whole screen)"
echo "======================================"

VARIANT="${VARIANT:-corrected}" python nbconverted/5b.single_cell_feature_select.py
exit_code=$?

if [ "$exit_code" -ne 0 ]; then
    echo "Single-cell feature selection FAILED (exit code $exit_code)"
    failed_plates+=("single_cell_feature_select:whole_screen")
else
    echo "Single-cell feature selection done for whole screen"
fi

# -----------------------------
# Step 4b-apply: corrected, normalized single cells (one plate/process at a time)
# Writes <screen profiles>/single_cell_profiles/<plate>_sc_<variant>_feature_selected.parquet: every QC-passed cell, the
# selected features only, no inverse normal transform. Single cells are never sphered.
# -----------------------------
echo "======================================"
echo "Step 4b-apply: corrected, normalized single cells (one plate/process at a time)"
echo "======================================"

variant="${VARIANT:-corrected}"
for plate_id in "${plate_ids[@]}"; do
    plate_key="${plate_id// /_}"
    echo ">>> Single-cell apply for ${plate_id}"
    STAGE="apply" PLATE_ID="$plate_key" OVERWRITE="$overwrite" VARIANT="$variant" python nbconverted/4b.single_cell_correct.py
    exit_code=$?

    if [ "$exit_code" -ne 0 ]; then
        echo "Single-cell apply FAILED for plate: ${plate_id} (exit code $exit_code)"
        failed_plates+=("single_cell_apply:${plate_id}")
    else
        echo "Single-cell apply done for plate: ${plate_id}"
    fi
done

deactivate

echo "======================================"
echo "Pipeline finished."
if [ ${#failed_plates[@]} -gt 0 ]; then
    echo "Failures:"
    printf '  %s\n' "${failed_plates[@]}"
    exit 1
fi

echo "✅ All steps completed successfully."
