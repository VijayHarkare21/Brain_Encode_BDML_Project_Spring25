# BrainAlign: Mapping Natural Modalities to EEG Representation Spaces

*Authors: Vijay Jayawant Harkare, Shubham Patel*  
*Department of Computer Science, New York University*

---

## Overview

**BrainAlign** introduces a novel framework that enhances visual and textual understanding in AI by aligning natural modality representations (from text and images) with human EEG-derived embedding spaces. Leveraging EEG foundation models and contrastive learning, BrainAlign bridges the gap between machine and human representations, improving AI performance in tasks such as image classification and sentiment analysis.

---

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

To download the required datasets:

* If using a SLURM-managed cluster:

  ```bash
  sbatch data_download.slurm
  ```

* Alternatively, manually download the datasets as specified in the `data_download.slurm` script.

## Results

The analyses aim to reveal the extent to which computational models can replicate brain-like representations. Visualizations and quantitative metrics are provided in the `figs/` directory and the respective notebooks.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.
