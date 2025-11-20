"""
Script for defining helpful functions.
"""

import pickle
from pathlib import Path
from typing import Any


def check_save_path(save_path: str | Path, allow_overwrite: bool = True, create_folder: bool = True) -> None:
    """
    Controls a save path.

    Checks that no file would be overwritten, and creates the folder to save into if it doesn't exist yet.
    """
    save_path = Path(save_path)
    if save_path.exists() and not allow_overwrite:
        raise FileExistsError(f"File {save_path} already exists and would be overwritten.")
    if create_folder:
        save_path.parent.mkdir(exist_ok=True)


def save_data(data, save_path: str | Path, label: str = "object") -> None:
    """
    Saves a Python object as a Pickle file.

    Saves the passed data with a given file name and prints a message.
    """
    save_path = Path(save_path)
    ext = save_path.suffix

    # Save in Pickle format and print message
    if ext != ".pkl":
        raise ValueError("The save path must have a pkl extension.")
    with open(save_path, "wb") as file:
        pickle.dump(data, file, protocol=-1)
    print(f"Saved {label} as an {ext} file to {save_path}.")


def load_data(file_path: str | Path) -> Any:
    """
    Loads a Python object from a Pickle file.
    """
    file_path = Path(file_path)

    # Load from Pickle format and return data
    ext = file_path.suffix
    if ext != ".pkl":
        raise ValueError("The file path must have a pkl extension.")
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data
