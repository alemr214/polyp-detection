from scripts.annotate_images import process_images

if __name__ == "__main__":
    IMAGE_FOLDER = ""
    IMAGE_EXT = ""
    MASK_FOLDER = ""
    MASK_EXT = ""
    OUTPUT_FOLDER = ""
    CLASS_INDEX = 0

    process_images(
        IMAGE_FOLDER, IMAGE_EXT, MASK_FOLDER, MASK_EXT, OUTPUT_FOLDER, CLASS_INDEX
    )
