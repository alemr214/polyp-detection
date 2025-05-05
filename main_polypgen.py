import os
import shutil
import numpy as np
import cv2 as cv
import os.path
import yaml
from scripts.process_images import (
    detect_object,
    normalize_coordiantes,
    yolo_format,
)
from scripts.manage_data import (
    create_dir,
)
# from google.colab.patches import cv2_imshow


def detect_imgs(infolder, ext=".tif"):
    items = os.listdir(infolder)

    flist = []
    for names in items:
        if names.endswith(ext) or names.endswith(ext.upper()):
            flist.append(os.path.join(infolder, names))

    return np.sort(flist)


# writing bbox
def save_bbox(txt_path, line):
    txt_path = txt_path + ".txt"
    with open(txt_path, "w") as myfile:
        myfile.write(line + "\n")  # append line


# Global variables
_EXT_FILE = ".jpg"  # image extension
_BASE_FOLDER = "data/clean"
_PATH_DATA = "data/raw/polypgen"
_NAME_DB = "polypgen"
_PATH_DATA_SEQ = _PATH_DATA + "/sequenceData/positive"

dbPolyps = _BASE_FOLDER + "/" + _NAME_DB
dirImages = dbPolyps + "/" + "images"
dirLabels = dbPolyps + "/" + "labels"
dirMasks = dbPolyps + "/" + "masks"

dirImagesTrain = dirImages + "/" + "train"
dirImagesVal = dirImages + "/" + "validation"
dirImagesTestSingle = dirImages + "/" + "test_single"
dirImagesTestSequence = dirImages + "/" + "test_sequence"

dirLabelsTrain = dirLabels + "/" + "train"
dirLabelsVal = dirLabels + "/" + "validation"
dirLabelsTestSingle = dirLabels + "/" + "test_single"
dirLabelsTestSequence = dirLabels + "/" + "test_sequence"

dirMasksTrain = dirMasks + "/" + "train"
dirMasksVal = dirMasks + "/" + "validation"
dirMasksTestSingle = dirMasks + "/" + "test_single"
dirMasksTestSequence = dirMasks + "/" + "test_sequence"

create_dir(dbPolyps)
create_dir(dirImages)
create_dir(dirLabels)

create_dir(dirImagesTrain)
create_dir(dirImagesVal)
create_dir(dirImagesTestSingle)
create_dir(dirImagesTestSequence)

create_dir(dirLabelsTrain)
create_dir(dirLabelsVal)
create_dir(dirLabelsTestSingle)
create_dir(dirLabelsTestSequence)

create_dir(dirMasksTrain)
create_dir(dirMasksVal)
create_dir(dirMasksTestSingle)
create_dir(dirMasksTestSequence)


def FindImages(
    subdir_list_center, path_data, ext_file, dir_Images, dir_Masks, dir_labels, type
):
    for centerIdx in range(len(subdir_list_center)):  # for each data center
        centerId = subdir_list_center[centerIdx]
        if type == "C":
            center = "/data_" + centerId
        else:
            center = "/" + centerId
        imageDir = path_data + center + "/images_" + centerId
        print(f"Procesando el centro {center}")
        allfileList = detect_imgs(imageDir, ext_file)
        maskDir = path_data + center + "/masks_" + centerId
        print("Imágenes que serán procesadas: " + str(len(allfileList)))
        # Recorre todas las imágenes procesadas en el centro actual
        for ii, imageFile in enumerate(allfileList[:]):
            "listimage files and find the type and modality of polyp"
            fileNameOnly = imageFile.split(os.sep)[-1].split(".")[0]
            image = cv.imread(imageFile)
            maskFile = maskDir + "/" + fileNameOnly + "_mask" + ext_file
            maskFile = maskFile.replace("]", "")
            "distinguish sizes of polyps and quantify numbers for each case"
            maskCordinates = detect_object(maskFile)
            maskCordinates = normalize_coordiantes(maskCordinates)
            line = ""
            if maskCordinates is not None:
                h, w, c = image.shape
                line = yolo_format(0, maskCordinates, w, h)
                shutil.copy(imageFile, dir_Images)
                shutil.copy(maskFile, dir_Masks)
                save_bbox(dir_labels + "/" + fileNameOnly, "\n".join(line))
                "Move image and create file"


SUBDIR_LIST_center = ["C1", "C2", "C3", "C4", "C5"]  # for training

FindImages(
    SUBDIR_LIST_center,
    _PATH_DATA,
    _EXT_FILE,
    dirImagesTrain,
    dirMasksTrain,
    dirLabelsTrain,
    "C",
)

SUBDIR_LIST = [
    "seq1",
    "seq2",
    "seq3",
    "seq4",
    "seq5",
    "seq6",
    "seq7",
    "seq8",
    "seq9",
    "seq10",
    "seq11",
    "seq12",
    "seq13",
    "seq14",
    "seq15",
]
FindImages(
    SUBDIR_LIST,
    _PATH_DATA_SEQ,
    _EXT_FILE,
    dirImagesTrain,
    dirMasksTrain,
    dirLabelsTrain,
    "S",
)

SUBDIR_LIST_center_Test = ["C6"]  # C6 is for testing
FindImages(
    SUBDIR_LIST_center_Test,
    _PATH_DATA,
    _EXT_FILE,
    dirImagesTestSingle,
    dirMasksTestSingle,
    dirLabelsTestSingle,
    "C",
)

SUBDIR_LIST_Test = [
    "seq16",
    "seq17",
    "seq18",
    "seq19",
    "seq20",
    "seq21",
    "seq22",
    "seq23",
]
FindImages(
    SUBDIR_LIST_Test,
    _PATH_DATA_SEQ,
    _EXT_FILE,
    dirImagesTestSequence,
    dirMasksTestSequence,
    dirLabelsTestSequence,
    "S",
)

allfileListTrain = detect_imgs(dirImagesTrain, _EXT_FILE)

shuffled_indices = np.random.permutation(len(allfileListTrain))
idx = int(0.8 * len(shuffled_indices))

training_idx = slice(0, idx)
training_data_tbl = allfileListTrain[shuffled_indices[training_idx]]

validation_idx = slice(training_idx.stop, len(shuffled_indices))
validation_data_tbl = allfileListTrain[shuffled_indices[validation_idx]]

allfileListTestSingle = detect_imgs(dirImagesTestSingle, _EXT_FILE)
allfileListTestSequence = detect_imgs(dirImagesTestSequence, _EXT_FILE)

# %%
print(
    "Total de imagénes de la base datos de pólipos: "
    + str(
        len(allfileListTrain)
        + len(allfileListTestSingle)
        + len(allfileListTestSequence)
    )
)
print("Elementos en el conjunto de entrenamiento     : " + str(len(allfileListTrain)))
print(
    "Seperación y generación del nuevo conjunto de entrenamiento  80%: "
    + str(len(training_data_tbl))
)
print(
    "Elementos en el conjunto de validación     20%: " + str(len(validation_data_tbl))
)
print(
    str(len(training_data_tbl))
    + " + "
    + str(len(validation_data_tbl))
    + " = "
    + str(len(training_data_tbl) + len(validation_data_tbl))
)

# %%
for ii, imageFile in enumerate(validation_data_tbl[:]):
    "listimage files and find the type and modality of polyp"
    fileNameOnly = imageFile.split(os.sep)[-1].split(".")[0]
    labelFile = dirLabelsTrain + "/" + fileNameOnly + ".txt"
    maskFile = dirMasksTrain + "/" + fileNameOnly + "_mask" + _EXT_FILE
    # move image
    shutil.move(imageFile, dirImagesVal)
    # movel label
    shutil.move(labelFile, dirLabelsVal)
    # move mask
    shutil.move(maskFile, dirMasksVal)
    print(str(ii) + "- archivo: " + fileNameOnly + " movido")
len(validation_data_tbl)

dataset_dir = dbPolyps


train_images_path = os.path.join(dataset_dir, "images/train")
validation_images_path = os.path.join(dataset_dir, "images/validation")
test_images_path = os.path.join(dataset_dir, "images/test_single")

class_names = ["polyp"]

data_yaml = {
    "path": dataset_dir,
    "train": train_images_path,
    "val": validation_images_path,
    "test": test_images_path,
    "nc": 1,
    "names": class_names,
}

yaml_path = dataset_dir + "/datasingle.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f)

print(f"Archivo YAML creado en {yaml_path}")

dataset_dir = dbPolyps


train_images_path = os.path.join(dataset_dir, "images/train")
validation_images_path = os.path.join(dataset_dir, "images/validation")
test_images_path = os.path.join(dataset_dir, "images/test_sequence")

class_names = ["polyp"]

data_yaml = {
    "path": dataset_dir,
    "train": train_images_path,
    "val": validation_images_path,
    "test": test_images_path,
    "nc": 1,
    "names": class_names,
}

yaml_path = dataset_dir + "/datasequence.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f)

print(f"Archivo YAML creado en {yaml_path}")
