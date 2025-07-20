#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition dllabdlc_gpu-rtx2080 --exclude=dlc2gpu01 # dllabdlc_gpu-rtx2080
#SBATCH --cpus-per-task=16

# Define a name for your job
#SBATCH --job-name nlp      # short: -J <job name>

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output outputs/logs/nlp%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error outputs/logs/nlp%x-%A.err    # STDERR  short: -e logs/%x-%A-job_name.out

# Define the amount of memory required per node
#SBATCH --gpus 1 

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate nlp

# Export CUDA debugging environment variables
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Running the job
start=`date +%s`

cd /work/dlclarge2/alipourn-nastaran/automl/nlp-classifier-automl

export PYTHONPATH=/home/alipourn/miniconda3/envs/nlp/bin/python

python run.py --data-path data --dataset ag_news --epochs 2 --data-fraction 0.2 --approach tfidf --output-path outputs/nlp-tfidf-agnews
end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
