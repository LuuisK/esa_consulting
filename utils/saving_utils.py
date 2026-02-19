"""Module with utilities for saving."""

import pickle
import warnings
from pathlib import Path
from typing import Any

import plotnine as p9
from plotnine.composition import Compose
from plotnine.exceptions import PlotnineWarning

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


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


def save_data(data: Any, save_path: str | Path) -> None:
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
    print(f"Saved file to {save_path}.")


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


def save_plot(plot: p9.ggplot | Compose, save_path: str | Path, **kwargs) -> None:
    """
    Saves a plotnine plot as a PNG file.

    Saves the passed data with a given file name and prints a message.
    """
    save_path = Path(save_path)
    ext = save_path.suffix

    # Save in PNG format and print message
    if ext != ".png":
        raise ValueError("The save path must have a png extension.")
    with warnings.catch_warnings():
        # Catch PlotnineWarnings that are issued every time a file is saved
        warnings.filterwarnings("ignore", category=PlotnineWarning)
        plot.save(filename=save_path, **kwargs)
    print(f"Saved file to {save_path}.")
