# Generating Brain-like Natural Modality Encodings

## Overview

This project explores the generation of brain-like natural modality encodings by integrating EEG data with computational models. It aims to bridge the gap between neural signals and machine learning representations, providing insights into how the brain processes information.

## Repository Structure

* **code/**: Contains the main Python scripts for data processing and model implementation.
* **figs/**: Stores figures and visualizations generated from the analyses.
* **rsa\_cca/**: Implements Representational Similarity Analysis (RSA) using Canonical Correlation Analysis (CCA) to compare EEG data with model embeddings.
* **rsa\_cca\_random/**: Serves as a control by applying RSA-CCA on randomized data to validate the significance of results.
* **test\_img\_labels.ipynb**: Jupyter notebook for testing image labels and their corresponding embeddings.
* **test\_zuco\_embeds.ipynb**: Notebook for analyzing embeddings from the ZuCo dataset, which includes EEG and eye-tracking data.
* **viz\_test.ipynb**: Contains visualization tests for the processed data and analysis results.
* **data\_download.slurm**: SLURM script for downloading necessary datasets in a high-performance computing environment.
* **eeg\_bdml\_env.txt** & **eeg\_env.yml**: Environment configuration files to set up the required dependencies using Conda.

## Getting Started

### Prerequisites

* Python 3.8 or higher
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

To download the required datasets:

* If using a SLURM-managed cluster:

  ```bash
  sbatch data_download.slurm
  ```

* Alternatively, manually download the datasets as specified in the `data_download.slurm` script.

## Usage

1. **Run analyses:**

   * Execute the Jupyter notebooks (`test_img_labels.ipynb`, `test_zuco_embeds.ipynb`, `viz_test.ipynb`) to perform data analysis and visualization.

2. **Perform RSA-CCA:**

   * Navigate to the `rsa_cca/` directory and run the scripts to compute the representational similarity between EEG data and model embeddings.

3. **Validate with Randomized Data:**

   * Use the `rsa_cca_random/` directory to perform control analyses with randomized data, ensuring the robustness of the RSA-CCA results.

## Results

The analyses aim to reveal the extent to which computational models can replicate brain-like representations. Visualizations and quantitative metrics are provided in the `figs/` directory and the respective notebooks.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.

## License

This project is licensed under the MIT License.
