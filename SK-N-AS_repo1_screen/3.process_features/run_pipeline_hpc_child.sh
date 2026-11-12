#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --account=amc-general
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=run_pipeline_child-%j.out

# activate preprocessing environment
module load miniforge
conda init bash
conda activate pccma_repo1_preprocessing_env

# prioritize the env's own libstdc++ over the system one in /lib64, which is
# older and missing symbols (e.g. GLIBCXX_3.4.29) required by libzmq.so.5
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# plate id passed as first argument
plate_id=$1
overwrite="${OVERWRITE:-}"

failed_steps=()

# Step 1: single-cell QC via papermill (skips if already done)
qc_output="./data/qc_results/${plate_id}_qc_annotations.parquet"
qc_notebook_dir="./executed_notebooks/1.single_cell_qc"
mkdir -p "$qc_notebook_dir"

if [ -f "$qc_output" ] && [ -z "$overwrite" ]; then
    echo "✅ ${plate_id} already QC'd (found ${qc_output})"
else
    papermill 1.single_cell_qc.ipynb "${qc_notebook_dir}/${plate_id}.ipynb" \
        -p plate_id "$plate_id" \
        -p render_diagnostics False
    if [ $? -ne 0 ]; then
        echo "QC FAILED for plate: $plate_id"
        failed_steps+=("qc")
    fi
fi

# Step 2: annotation
STEPS=annotate PLATE_ID="$plate_id" OVERWRITE="$overwrite" python nbconverted/2.bulk_profiles_SKNAS.py
if [ $? -ne 0 ]; then
    echo "Annotation FAILED for plate: $plate_id"
    failed_steps+=("annotate")
fi

# Step 3: aggregation (per-plate only -- the whole-screen steps that follow,
# the counts and pool steps of 2.bulk_profiles_SKNAS.ipynb, are not run here; run
# them once every plate's aggregated profile is in)
STEPS=aggregate PLATE_ID="$plate_id" OVERWRITE="$overwrite" python nbconverted/2.bulk_profiles_SKNAS.py
if [ $? -ne 0 ]; then
    echo "Aggregation FAILED for plate: $plate_id"
    failed_steps+=("aggregate")
fi

# Steps after this need the whole-screen tables from the counts and pool steps (STEPS=counts, then STEPS=pool,correct,normalize), so they are
# not run here. Once every plate's aggregate is in, run those two once, then submit
# run_single_cell_hpc_child.sh per plate with STAGE=stats, run 5b.single_cell_feature_select once, and submit it
# again per plate with STAGE=apply.

conda deactivate

if [ ${#failed_steps[@]} -gt 0 ]; then
echo "Plate $plate_id FAILED at steps: ${failed_steps[*]}"
exit 1
fi

echo "Plate $plate_id completed all steps successfully."
