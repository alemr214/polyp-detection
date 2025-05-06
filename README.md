# Polyp-Detection

A deep learning project for automatic detection and localization of polyps in endoscopic images. This repository includes scripts necessary to work for data preparation, model training, inference scripts, and evaluation metrics for robust performance analysis.

## Table of Contents

* [Features](#features)
* [Getting Started](#getting-started)

  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Project Structure](#project-structure)
* [Contributing](#contributing)
* [License](#license)

## Features

* **Data preparation**: Tools to preprocess and augment endoscopic images.
* **Model architectures**: Implementations of state-of-the-art object detection models (e.g., YOLOv11).
* **Training scripts**: Easy-to-use Python scripts to train models on custom datasets.
* **Inference pipeline**: Run inference on single images, directories, or video streams.
* **Evaluation metrics**: Compute precision, recall, F1-score, and mAP for model performance.

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
polyp-detection/          # Root directory
├── configs/              # Configuration files (YAML)
├── data/                 # Dataset directory
├── runs/                 # Training, exporting and validation outputs
├── scripts/              # Python functions
├── main_polypgen.py      # Polypgen working 
├── main.py               # Main file to run the scripts
├── requirements.txt      # Python dependencies
└── README.md             # Project overview
```

## Contributing

Contributions are welcome! Please adhere to the following:

1. Fork the repository and create a feature branch: `git checkout -b feature-name`
2. Commit your changes: \`git commit -m "Add some feature"
3. Push to your fork: `git push origin feature-name`
4. Open a pull request on GitHub and describe your changes.

Please follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
