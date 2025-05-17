# BrainAlign: Mapping Natural Modalities to EEG Representation Spaces

*Authors: Vijay Jayawant Harkare, Shubham Patel*  
*Department of Computer Science, New York University*

---

## Overview

**BrainAlign** introduces a novel framework that enhances visual and textual understanding in AI by aligning natural modality representations (from text and images) with human EEG-derived embedding spaces. Leveraging EEG foundation models and contrastive learning, BrainAlign bridges the gap between machine and human representations, improving AI performance in tasks such as image classification and sentiment analysis.

---

## Getting Started

### Prerequisites

* Python 3.11
* Conda for environment management

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/VijayHarkare21/Brain_Encode_BDML_Project_Spring25.git
   cd Brain_Encode_BDML_Project_Spring25
   ```

2. **Set up the Conda environment:**

   ```bash
   conda env create -f eeg_env.yml
   conda activate eeg_env
   ```

### Data Acquisition

#### THINGS-EEG

* Use [https://osf.io/3jk45/](https://osf.io/3jk45/) as the base link for downloading all required data from the THINGS-EEG dataset.
* Download the image set into `datasets/THINGS-EEG/image_set`.

  * Place training images in `image_set/training_images`.
  * Place test images in `image_set/test_images`.
* Download the following files and place them in `image_set` (download from the main THINGS database and the THINGS-EEG link above):

  * `category27_bottom-up.tsv`
  * `category27_manual.tsv`
  * `category27_top-down.tsv`
  * `category53_wide-format.tsv`
  * `image_metadata.npy`
* Download and unzip all raw EEG files into `datasets/THINGS-EEG/eeg_data/raw_data`.
* Clone the eeg\_encoding repository (into the `code` folder):

  ```bash
  git clone https://github.com/gifale95/eeg_encoding code/eeg_encoding
  ```

  * Update `code/eeg_encoding/02_eeg_preprocessing/preprocessing_utils.py` to resample at 200 Hz (instead of 100 Hz), leaving all other options the same.
  * Preprocess the raw EEG data (this may take some time). After processing, preprocessed signals will be available at `datasets/THINGS-EEG/eeg_dataset/preprocessed_data`.

#### ZuCo 2.0

* Use [https://osf.io/2urht/](https://osf.io/2urht/) as the base link for downloading all required data from the ZuCo 2.0 dataset.
* Download the following folders into `datasets/ZuCo2/2urht/osfstorage`, maintaining the structure:

  * `scripts`
  * `task1 – NR/Matlab files`
  * `task_materials`

### Embedding Creation for Preliminary Experiments

#### Clone Model Repositories

* Clone LaBraM and CBraMod repositories (into the `code` folder):

  ```bash
  git clone https://github.com/935963004/LaBraM code/LaBraM
  git clone https://github.com/wjq-learning/CBraMod code/CBraMod
  ```

* The current repository contains additional files for LaBraM embedding generation in `code/LaBraM`. Place these files in the cloned LaBraM repository.

* Download and place pretrained weights in respective directories, as instructed in the respective repositories.

#### ZuCo 2.0 Embedding Preparation

* Clone the EEG-To-Text repository (into the `code` folder):

  ```bash
  git clone https://github.com/MikeWangWZHL/EEG-To-Text code/EEG-To-Text
  ```
* Copy the files from `code/EEG-To-Text/utils` to the respective folder.
* Adjust the paths in `code/EEG-To-Text/utils/get_data_in_npy.py`, then run it to obtain the data in numpy file format (needed for LaBraM and CBraMod embeddings).
* Run:

  ```bash
  python code/cbramod_embeddings_generator.py
  ```

  to generate CBraMod EEG embeddings for text instances (ensure all required directories exist).
* After this, run:

  ```bash
  python code/LaBraM/labram_embeddings_generator.py
  ```

  to generate LaBraM EEG embeddings for text instances.

#### THINGS-EEG Embedding Preparation

* Run:

  ```bash
  python code/cbramod_embeddings_generator_image.py
  ```

  to generate CBraMod EEG embeddings for image instances.
* After this, run:

  ```bash
  python code/LaBraM/labram_embeddings_generator_image.py
  ```

  to generate LaBraM EEG embeddings for image instances.

### Running Analyses on Representation Spaces

#### RSA, CCA, Regression, and Other Alignment Analyses

* To perform RSA and CCA analyses:

  ```bash
  python code/rsa_cca_grouped_analysis_with_ensemble.py --mode both --eeg_image_dir path_to/datasets/THINGS-EEG/eeg_dataset/embeds --eeg_text_dir path_to/datasets/ZuCo2/2urht/osfstorage/task1\ -\ NR/npy_file/embeds --output_csv path_to/rsa_cca_random/results_rsa_cca_avg_both_all.csv --figures_dir path_to/rsa_cca_random/figs_rsa_cca --subject_aggregate average --batch_size 64 --embed_cache_dir path_to/nat_mod_embeds/ --run_ensemble
  ```

* For CCA train/test regression analyses:

  ```bash
  python code/cca_validation_pipeline_with_ensemble.py --mode both --eeg_image_dir path_to/datasets/THINGS-EEG/eeg_dataset/embeds --eeg_text_dir path_to/datasets/ZuCo2/2urht/osfstorage/task1\ -\ NR/npy_file/embeds --output_csv path_to/rsa_cca_random/results_cca_pipeline_avg_both_all.csv --figures_dir path_to/rsa_cca_random/figs_cca_pipeline --subject_aggregate average --batch_size 64 --embed_cache_dir path_to/nat_mod_embeds/ --test_size 0.2 --run_ensemble
  ```

* For extra alignment analyses:

  ```bash
  python code/final_extra_validation_tests.py --mode both --eeg_image_dir path_to/datasets/THINGS-EEG/eeg_dataset/embeds --eeg_text_dir path_to/datasets/ZuCo2/2urht/osfstorage/task1\ -\ NR/npy_file/embeds --output_csv path_to/rsa_cca_random/results_final_extra_vals_avg_all.csv --figures_dir path_to/rsa_cca_random/figs_extra_vals --subject_aggregate average --batch_size 64 --embeds_cache_dir path_to/nat_mod_embeds/ --test_size 0.2 --run_ensemble --cache_dir path_to/cache
  ```

* To run control analyses on random noise, run the same-named files from `code/random_noise_prelim_experiments` with similar options.

### EEG Foundation Model and Natural Modality Classifiers

#### EEG Foundation Model Classifiers

* To run EEG embedding classifiers:

  ```bash
  python code/classifiers/classifiers_eeg_embeds.py --mode image --eeg_dir none --embedding_type both --subject_handling stack --classifier_type all --output_dir path_to/code/classifiers/results --image_metadata path_to/datasets/THINGS-EEG/image_set/image_metadata.npy --text_csv path_to/code/classifiers/data/combined_data_with_labels.csv --image_eeg_dir path_to/datasets/THINGS-EEG/eeg_dataset/embeds --text_eeg_dir path_to/datasets/ZuCo2/2urht/osfstorage/task1\ -\ NR/npy_file/embeds_only --scale_features --things_map_path path_to/datasets/THINGS-EEG/image_set/category27_top-down.tsv
  ```

#### Natural Modality Classifiers

* To run the six classifiers on natural modality embeddings (e.g., ResNet-50, CLIP, ViT), use the notebooks:

  * `code/classfiers/natural_modes/train_image_classifiers.ipynb`
  * `code/classfiers/natural_modes/train_text_classifiers.ipynb`

### Jointly Training Shared Latent Space Learner Models

* For training the latent space learner for images:

  ```bash
  python code/joint_projector_cbramod_resnet_image.py --eeg_dir path_to/datasets/THINGS-EEG/eeg_dataset/preprocessed_data --eeg_ckpt path_to/code/CBraMod/pretrained_weights/pretrained_weights.pth --model_dir path_to/code/finetune_eeg_embed_models/finetuned_model/latest --out path_to/code/finetune_eeg_embed_models/finetuned_model/latest/best_cbramod_image_model_cpu_new.pth --batch 512 --metadata path_to/datasets/THINGS-EEG/image_set/image_metadata.npy --things_map path_to/datasets/THINGS-EEG/image_set/category27_top-down.tsv --img_parent_dir path_to/datasets/THINGS-EEG/image_set/training_images
  ```

* For training the latent space learner for text:

  ```bash
  python code/joint_projector_cbramod_bert_text.py --eeg_dir path_to/datasets/ZuCo2/2urht/osfstorage/task1\ -\ NR/npy_file/embeds --eeg_ckpt path_to/code/CBraMod/pretrained_weights/pretrained_weights.pth --model_dir path_to/code/finetune_eeg_embed_models/finetuned_model/latest --out path_to/code/finetune_eeg_embed_models/finetuned_model/latest/best_cbramod_text_model_cpu.pth --batch 16 --label_csv path_to/code/classifiers/data/combined_data_with_labels.csv
  ```

---

## Results

The analyses aim to reveal the extent to which computational models can replicate brain-like representations. Results are organized across several directories:

### Visualization Results
- **`figs/`**: Contains visualizations of EEG foundation model representation space clusters using t-SNE and UMAP dimensionality reduction techniques. It also includes montage plots showing the EEG channels used throughout all analyses

### Representation Analysis Results
- **`rsa_cca/`**: Contains results from Representational Similarity Analysis (RSA), Canonical Correlation Analysis (CCA), and additional validation tests
- **`rsa_cca_random/`**: Contains control analysis results using random noise for validation of the primary findings

### Classification Results
- **`code/classifiers/results/`**: Contains performance metrics and evaluation results for text and image classification tasks performed using EEG foundation models
- **`code/classifiers/natural_modes/`**: Contains performance metrics for text and image classification tasks using natural modality feature extractors (ResNet-50, CLIP, ViT, etc.)

### Model Training Results
- **`code/finetune_eeg_embed_models/finetuned_model/latest/`**: Contains checkpoints and performance metrics from joint training of the shared latent space learning models for both image and text modalities

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.
