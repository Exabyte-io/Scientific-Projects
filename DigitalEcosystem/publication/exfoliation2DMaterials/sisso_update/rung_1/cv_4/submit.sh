#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=4
#SBATCH --time=4:00:00
#SBATCH --partition=p.talos
#SBATCH --job-name=exabyte_cv_4
#SBATCH --output=exabyte-%j.out
#SBATCH --error=exabyte-%j.error


ulimit -s unlimited
export LD_LIBRARY_PATH=$I_MPI_ROOT/intel64/lib/:$I_MPI_ROOT/intel64/lib/release/:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# For pinning threads correctly:
export OMP_PLACES=cores

srun /u/tpurcell/git/cpp_sisso/bin/sisso++
sstat  -j   $SLURM_JOB_ID.batch   --format=JobID,MaxVMSize,MaxRSS,MaxRSSNode

