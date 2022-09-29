#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition=p.talos
#SBATCH --job-name=bandgap-tpot
#SBATCH --output=bandgap-tpot-%j.out
#SBATCH --error=bandgap-tpot-%j.error

ulimit -s unlimited

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# For pinning threads correctly:
export OMP_PLACES=cores

conda activate exabyte_wf

python run_tpot.py
sstat  -j   $SLURM_JOB_ID.batch   --format=JobID,MaxVMSize,MaxRSS,MaxRSSNode

