#!/bin/bash
#SBATCH --account=csci_ga_3033_077-2025sp
#SBATCH --partition=g2-standard-12
#SBATCH --job-name=cbramod_embeddings_image
#SBATCH --open-mode=append
#SBATCH --output=/scratch/vjh9526/bdml_2025/project/%j_%x.out
#SBATCH --error=/scratch/vjh9526/bdml_2025/project/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec --bind /scratch/ $nv --overlay /scratch/vjh9526/bdml_2025/project/env_files/tmp/overlay-50G-10M.ext3:ro /scratch/vjh9526/bdml_2025/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif /bin/bash -c "
source /ext3/env.sh
conda activate eeg_env
cd /scratch/vjh9526/bdml_2025/project/code
python cbramod_embeddings_generator_image.py
exit
"