# Polyp Detector

YOLOv11 pre-trained model to detect colo-rectal polyps using public datasets.

## Getting Started

All code below is written in Python 3.12.8 version, you'll find all libraries required in the requirements.txt file, the model pre-trained is the CNN model YOLOv11.

### Folder structure

    .
    ├── scripts/                        # Automate task with functions
    │   ├── annotate_images.py          # Annotate images in YOLO format
    │   └── manage_data.py              # Utils functions as copy, rename, etc
    ├── .gitignore                      # Ignore files
    ├── LICENSE                         # Apache 2.0 License
    ├── README.md                       # README file
    ├── main.py                         # Main file to run the program
    └── requirements.txt                # Libraries to use

### Prerequisites

Requirements for the software and other tools to build.

- [YOLOv11](https://docs.ultralytics.com/models/yolo11/)

#### Libraries

Found in the [requirements.txt](requirements.txt) file.

| Library             | Version            |
| ------------  | ------------ |
| ultralytics      | 8.3.65        |
| numpy      | 1.26.4       |
| torch       | 2.5.1       |
| torchaudio      | 2.5.1         |
| torchvision       | 0.20.1      |
| opencv-python  | 4.11.0.86     |
| matplotlib    | 3.10.0        |

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
