# Kaggle Competition Starter Notebook

Welcome to our **2025 Kaggle Competition of AI Applied to Medicine at UC3M** repository! 

This GitHub repository contains the **starter Jupyter Notebook** designed to help you get up and running with the following:

1. **Data Loading**: Shows how to load images from HDF5 files (`train-image.hdf5` and `test-image.hdf5`) and read the accompanying metadata.
2. **EDA & Preprocessing**: Performs a minimal exploratory data analysis (EDA) and basic preprocessing (resizing, normalization, etc.).
3. **Modeling**: Demonstrates how to load a **pretrained ResNet50**, then optionally fine-tune it (placeholder code for illustration).
4. **Submission Generation**: Creates a `submission.csv` file with the **required format** (`isic_id,target`) for the competition on Kaggle.

## Quick Start

1. **Clone this repo** or download the Jupyter Notebook (`.ipynb` file).
2. **Open** the notebook in Jupyter (e.g., JupyterLab or Jupyter Notebook) or directly in Kaggle’s environment.
3. **Follow the cells** step by step—each section guides you through:
   - Loading data from the `.csv` files and HDF5 images.
   - Setting up PyTorch datasets, transforms, and data loaders.
   - (Optional) Training or fine-tuning a pretrained model.
   - Generating predictions for the test set and creating a valid `submission.csv`.

## How to Use

- **Install Dependencies**: Make sure you have Python 3.x, `pandas`, `numpy`, `torch`, `torchvision`, `opencv-python`, `matplotlib`, and `h5py` installed.
- **Data Access**: Place the HDF5 files and metadata CSVs in the appropriate directories as indicated by the notebook.
- **Customize**: Adjust hyperparameters, training loops, and transforms to improve your model’s performance.
- **Submit**: Upload your `submission.csv` to the internal Kaggle competition page to see how your solution ranks!

## Contributing

Feel free to **fork** this repository and tailor the notebook to your own approach. If you have improvements or useful scripts, consider sharing them via **pull requests** so everyone can learn.

Good luck with the competition!
