# Polyp-Detection

A deep learning project for automatic detection and localization of polyps in endoscopic images. This repository includes scripts necessary to work for data preparation, model training, inference scripts, and evaluation metrics for robust performance analysis.

## Table of Contents

* [Features](#features)
* [Datasets used](#datasets-used)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Project Structure](#project-structure)
* [License](#license)

## Features

* **Data preparation**: Tools to preprocess and augment endoscopic images.
* **Model architectures**: Implementations of state-of-the-art object detection models (e.g., YOLOv11).
* **Training scripts**: Easy-to-use Python scripts to train models on custom datasets.
* **Inference pipeline**: Run inference on single images, directories, or video streams.
* **Evaluation metrics**: Compute precision, recall, F1-score, and mAP for model performance.

## Datasets used

* [PolypGen](https://www.synapse.org/#!Synapse:syn45200214)
* [CVC-Clinic DB](https://www.kaggle.com/datasets/balraj98/cvcclinicdb/data)
* [CVC-Colon DB](https://www.kaggle.com/datasets/longvil/cvc-colondb)
* [ETIS-LaribPolypDB](https://www.kaggle.com/datasets/nguyenvoquocduong/etis-laribpolypdb)
* [Kvasir-SEG](https://www.kaggle.com/datasets/debeshjha1/kvasirseg)
* [Kvasir-Sessile](https://www.kaggle.com/datasets/debeshjha1/kvasirsessile)

## Getting Started

Follow these instructions to set up a local development environment and run experiments.

### Prerequisites

* Python 3.12
* PyTorch 2.7
* CUDA 10.2 or higher (for GPU acceleration)
* Git

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/alemr214/polyp-detection.git
   cd polyp-detection
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\\Scripts\\activate`
   ```

3. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
polyp-detection/                    #Root
├── configs/                        # Configuration files (YAML)
│   ├── dataset_name/
│   │   └── dataset.yaml
│   └── ...
├── data/                           # Dataset directory
│   ├── clean/                      # Clean images processed with process_images.py
│   │   ├── dataset_name/
│   │   │   ├── bbox/               # Bounding Boxes
│   │   │   │   ├── image.png
│   │   │   │   └── ...
│   │   │   ├── images/             # Splited images
│   │   │   │   ├── test/
│   │   │   │   │   ├── image.png
│   │   │   │   │   └── ...
│   │   │   │   ├── train/
│   │   │   │   │   ├── image.png
│   │   │   │   │   └── ...
│   │   │   │   └── val/
│   │   │   │       ├── image.png
│   │   │   │       └── ...
│   │   │   ├── labels/             # Split labels
│   │   │   │   ├── test/
│   │   │   │   │   ├── image.txt
│   │   │   │   │   └── ...
│   │   │   │   ├── train/
│   │   │   │   │   ├── image.txt
│   │   │   │   │   └── ...
│   │   │   │   └── val/
│   │   │   │       ├── image.txt
│   │   │   │       └── ...
│   │   │   └── masks/              # Binary masks
│   │   │       ├── image.png
│   │   │       └── ...
│   │   └── ...
│   └── raw/
│       └── dataset_name/
│           └── ...
├── runs/                           # Training, exporting and predictions
│   ├── predict/
│   │   └── ...
│   └── train/
│       └── ...
├── scripts/                        # Python functions
│   ├── evaluate_datasets.py
│   ├── manage_data.py
│   ├── process_images.py
│   └── yolo_utils.py
├── main.py                         # Main file to run the scripts
├── main_polypgen.py                # Script to process polypgen dataset
├── requirements.txt                # Python dependencies
└── endomind_advanced.pt            # Endomind model
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
