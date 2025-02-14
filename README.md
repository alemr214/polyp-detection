# Polyp Detector

YOLOv11 pre-trained model to detect colo-rectal polyps using public datasets.

## Getting Started

All code below is written in Python 3.12.8 version, you'll find all libraries required in the requirements.txt file, the model pre-trained is the CNN model YOLOv11.

### Folder structure

    .
    ├── data/                                           # Base path to all datasets
    │   └── clean/                                      # Clean path for cleaned data
    │       ├── cvc_clinic_db/                          # Name of datasets
    │       │   ├── bbox                                # Images with its bounding boxes drawn with the binary mask image
    │       │   ├── images/                             # Path for images and split in YOLO format
    │       │   │   ├── test/                           
    │       │   │   ├── train/
    │       │   │   └── val/
    │       │   ├── labels                              # Path for labels and split in YOLO format
    │       │   └── masks                               # Binary masks
    │       ├── cvc_colon_db/
    │       │   └── ...
    │       ├── etis_laribpolypdb/
    │       │   └── ...
    │       ├── kvasir_seg/
    │       │   └── ...
    │       ├── sessile_main_kvasir_seg/
    │       │   └── ...
    │       └── predict_images/                         # Images for predict feature functions
    ├── scripts/
    │   ├── manage_data.py                              # Script for manage images and utilities
    │   └── process_images.py                           # Process images and annotate functions
    ├── data_preparation.py                             # Base script to prepare images and other files
    ├── train_model.py                                  # Ultralytics functions to use and train YOLO model
    ├── README.md                                       
    ├── LICENSE
    └── requirements.txt

### Prerequisites

Requirements for the software and other tools to build.

- [YOLOv11](https://docs.ultralytics.com/models/yolo11/)

#### Libraries

Found in the [requirements.txt](requirements.txt) file.

#### Datasets

- [PolypGen](https://www.synapse.org/#!Synapse:syn45200214)
- [CVC-Clinic DB](https://www.kaggle.com/datasets/balraj98/cvcclinicdb/data)
- [CVC-Colon DB](https://www.kaggle.com/datasets/longvil/cvc-colondb)
- [ETIS-LaribPolypDB](https://www.kaggle.com/datasets/nguyenvoquocduong/etis-laribpolypdb)
- [Kvasir-SEG](https://www.kaggle.com/datasets/debeshjha1/kvasirseg)
- [Kvasir](https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset)
- [Kvasir-Sessile](https://www.kaggle.com/datasets/debeshjha1/kvasirsessile)
- [Hyper Kvasir](https://www.kaggle.com/datasets/kelkalot/the-hyper-kvasir-dataset/data)

### Installing

To work with the libraries used in the project, you should run the following:

> Recommendation: create a virtual environment to work, you can follow the guide in the link: [Guide venv](https://docs.python.org/3/library/venv.html)

Upgrade pip before install libraries

    pip install --upgrade pip

Install the libraries used on

    pip install -r requirements.txt

## Authors

- **Alemr214** - *Main Developer* -
    [alemr214](https://github.com/alemr214)

## License

This project is licensed under the [Apache License](LICENSE)
