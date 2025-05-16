#!/bin/bash
#SBATCH --account=csci_ga_3033_077-2025sp
#SBATCH --partition=n2c48m24
#SBATCH --job-name=rsa_cca_validations_text
#SBATCH --open-mode=append
#SBATCH --output=/scratch/vjh9526/bdml_2025/project/%j_%x.out
#SBATCH --error=/scratch/vjh9526/bdml_2025/project/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=05:00:00
#SBATCH --requeue

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

singularity exec --bind /scratch/ $nv --overlay /scratch/vjh9526/bdml_2025/project/env_files/tmp/overlay-50G-10M.ext3:ro /scratch/vjh9526/bdml_2025/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif /bin/bash -c "
source /ext3/env.sh
conda activate eeg_env
cd /scratch/vjh9526/bdml_2025/project/code
python rsa_cca_grouped_analysis_with_ensemble.py --mode text --eeg_image_dir "/scratch/vjh9526/bdml_2025/project/datasets/THINGS-EEG/eeg_dataset/embeds" --eeg_text_dir /scratch/vjh9526/bdml_2025/project/datasets/ZuCo2/2urht/osfstorage/task1\ -\ NR/npy_file/embeds --output_csv "/scratch/vjh9526/bdml_2025/project/rsa_cca/results_rsa_cca.csv" --figures_dir "/scratch/vjh9526/bdml_2025/project/rsa_cca/figs/" --subject_aggregate stack --batch_size 256 --embed_cache_dir "/scratch/vjh9526/bdml_2025/project/nat_mod_embeds/" --run_ensemble
exit
"