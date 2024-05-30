from typing import Dict
import json
import os


def check_JSON_file_path_integrity(file_path: str) -> None:
    """
    Check if the given file path is valid for a JSON file.

    This function ensures that the provided file path ends with a '.json'
    extension and that the directory containing the file exists.

    Args:
        file_path (str): The path to the JSON file.

    Raises:
        AssertionError: If the file path does not end with '.json' or
                        if the directory does not exist.
    """
    assert file_path.endswith(".json"), "File path must end with .json"
    assert os.path.isdir(
        os.path.dirname(file_path)
    ), f"Directory {os.path.dirname(file_path)} not exist"


def read_json_file(file_path: str) -> Dict:
    """
    Read data from a JSON file.

    This function reads and returns the data from a JSON file located at the
    specified file path.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        Dict: The data read from the JSON file.

    Raises:
        AssertionError: If the file path does not pass integrity checks.
        FileNotFoundError: If the file does not exist.
        JSONDecodeError: If the file is not a valid JSON.
    """
    check_JSON_file_path_integrity(file_path)

    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def write_json_file(data: Dict, file_path: str) -> None:
    """
    Write data to a JSON file.

    This function writes the given data to a JSON file at the specified file
    path.

    Args:
        data (Dict): The data to be written to the JSON file.
        file_path (str): The path to the JSON file.

    Raises:
        AssertionError: If the file path does not pass integrity checks.
    """
    check_JSON_file_path_integrity(file_path)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
