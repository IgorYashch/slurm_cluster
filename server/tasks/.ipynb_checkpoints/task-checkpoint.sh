#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=result-%j.out
#SBATCH --time=1:00

echo "Starting job $SLURM_JOB_ID"
sleep 10
echo "Job $SLURM_JOB_ID completed"
