#!/bin/bash
#SBATCH --account=csci_ga_3033_077-2025sp
#SBATCH --partition=n2c48m24
#SBATCH --job-name=cca_pipeline_validations
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
python cca_validation_pipeline_with_ensemble.py --mode both --eeg_image_dir "/scratch/vjh9526/bdml_2025/project/datasets/THINGS-EEG/eeg_dataset/embeds" --eeg_text_dir /scratch/vjh9526/bdml_2025/project/datasets/ZuCo2/2urht/osfstorage/task1\ -\ NR/npy_file/embeds --output_csv "/scratch/vjh9526/bdml_2025/project/rsa_cca/results_cca_pipeline_avg_both.csv" --figures_dir "/scratch/vjh9526/bdml_2025/project/rsa_cca/figs/" --subject_aggregate average --batch_size 128 --embed_cache_dir "/scratch/vjh9526/bdml_2025/project/nat_mod_embeds/" --test_size 0.2 --run_ensemble
exit
"