import yaml


def create_yaml_file(
    base_dataset_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    number_classes: int,
    classes_names: list[str],
    yaml_file_path: str,
) -> None:
    """
    Create a YAML file with the dataset information.

    Args:
        base_dataset_path (str): Dataset path where the train, val and test folders are located.
        train_path (str): Relative path to the base folder.
        val_path (str): Relative path to the base folder.
        test_path (str): Relative path to the base folder.
        number_classes (int): number of classes in the dataset.
        classes_names (list[str]): List with the classes names.
        yaml_file_path (str): Path to save the YAML file.
    """
    data = {
        "path": base_dataset_path,
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": number_classes,
        "names": classes_names,
    }

    with open(yaml_file_path + "dataset.yaml", "w") as file:
        yaml.dump(data, file)

    print(f"YAML file created at {yaml_file_path}")
