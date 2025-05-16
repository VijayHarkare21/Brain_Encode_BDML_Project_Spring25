#!/bin/bash
#SBATCH --account=csci_ga_3033_077-2025sp
#SBATCH --partition=n2c48m24
#SBATCH --job-name=data_download
#SBATCH --open-mode=append
#SBATCH --output=/scratch/vjh9526/bdml_2025/project/%j_%x.out
#SBATCH --error=/scratch/vjh9526/bdml_2025/project/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=7:00:00
#SBATCH --requeue

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec --bind /scratch/ --overlay /scratch/vjh9526/bdml_2025/project/env_files/overlay-50G-10M.ext3:ro /scratch/vjh9526/bdml_2025/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif /bin/bash -c "
source /ext3/env.sh
conda activate eeg_env
cd /scratch/vjh9526/bdml_2025/project/
osf -p 2urht clone
exit
"