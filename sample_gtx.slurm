#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Maverick2 GTX nodes
#----------------------------------------------------

#SBATCH -J myjob                        # Job name
#SBATCH -o myjob.o%j                    # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e myjob.e%j                    # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gpu-a100                     # Queue (partition) name
#SBATCH -N 1                            # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                            # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:10:00                     # Run time (hh:mm:ss)
#SBATCH --mail-user=madhan.sanikommu@utexas.edu
#SBATCH --mail-type=all                 # Send email at begin and end of job (can assign begin or end as well)
## SBATCH -A Hardware-Acceleratio         # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...

# Launch code...
# module load intel/17.0.4 python3/3.6.3
# module load cuda/10.0 cudnn/7.6.2 nccl/2.4.7
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/10.0/lib64
module load python3
module --ignore_cache load "intel/19.1.1"
module --ignore_cache load "cuda/11.4"
# module load cudnn nccl
source $WORK/hmlproject_virtualenv/bin/activate

python3 $WORK/hmlproject/lib/getgpuspecs.py > $WORK/hmlproject/out.txt

# python3 $WORK/hmlproject/lib/getgpuspecs.py > $WORK/hmlproject/gpuspecs.txt
# python3 $WORK/hmlproject/roofline2.py > $WORK/hmlproject/roofline2_output.txt

# ---------------------------------------------------