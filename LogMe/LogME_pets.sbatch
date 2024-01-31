#!/bin/sh
#SBATCH --partition=general 
#SBATCH --qos=long
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128240
#SBATCH --chdir=/tudelft.net/staff-umbrella/zlitransfer/LogME

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

# Use this simple command to check that your sbatch settings are working (it should show the GPU that you requested)
/usr/bin/nvidia-smi

# Your job commands go below here

# Uncomment these lines and adapt them to load the software that your job requires
module use /opt/insy/modulefiles
module --ignore_cache load cuda/12.1 cudnn/12-8.9.1.23
module load miniconda/3.8
conda activate /tudelft.net/staff-umbrella/zlitransfer/env-finetune

# Your job commands go above here
# Set HuggingFace default path
export HF_HOME=/tudelft.net/staff-umbrella/zlitransfer/hf/misc
export HF_DATASETS_CACHE=/tudelft.net/staff-umbrella/zlitransfer/hf/datasets
export TRANSFORMERS_CACHE=/tudelft.net/staff-umbrella/zlitransfer/hf/models

# Get configuration from filename and run.
current_filename=$SLURM_JOB_NAME
sbatch_file_name="${current_filename%.sbatch}"

METHOD="${sbatch_file_name%%_*}"
DATASET_NAME=${sbatch_file_name#*_}
OUTPUT="../out/${METHOD}/log_leep_${DATASET_NAME}.out"

echo "Method: ${METHOD}"
echo "Dataset: ${DATASET_NAME}"
echo "Out file: ${OUTPUT}"
srun --output=${OUTPUT} python3 compute.py --dataset_name ${DATASET_NAME} --method=${METHOD}

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
