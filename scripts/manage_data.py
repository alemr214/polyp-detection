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
