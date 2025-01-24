import shutil, os


def create_dir(output_path: str) -> None:
    """
    Create directory if no exists

    Args:
        output_path (str): Directory output_path
    """
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    except OSError:
        print(f"Error: Creating directory. {output_path}")


def copy_images(source_path: str, output_path: str, image_ext: str) -> None:
    """
    Move images to a new directory

    Args:
        source_path (str): Directory path to images
        image_ext (str): Image extension
    """
    create_dir(output_path)  # Create output directory
    for file in os.listdir(source_path):
        if file.endswith(image_ext):
            source_path_ = os.path.join(source_path, file)
            output_path_ = os.path.join(output_path, file)
            try:
                shutil.copy(source_path_, output_path_)
            except Exception as e:
                print(f"Error: Moving file. {source_path} {file}: {e}")
