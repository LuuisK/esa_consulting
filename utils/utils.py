"""
Script for defining helpful functions.
"""

import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotnine as p9


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


def save_data(data: Any, save_path: str | Path, label: str = "object") -> None:
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


def save_plot(plot: p9.ggplot, save_path: str | Path, label: str = "plot", dpi: int | None = None, **kwargs) -> None:
    """
    Saves a ggplot plot as a PNG file.

    Saves the passed data with a given file name and prints a message.
    """
    save_path = Path(save_path)
    ext = save_path.suffix

    # Save in PNG format and print message
    if ext != ".png":
        raise ValueError("The save path must have a png extension.")
    plot.save(filename=save_path, dpi=dpi, **kwargs)
    print(f"Saved {label} as an {ext} file to {save_path}.")


def turn_cats_to_continuous(data: pd.DataFrame, treat_cat_as_continuous: Iterable[str] | None) -> pd.DataFrame:
    """
    Turn specified variables of data into continuous ones.

    Used for categorical (pandas.CategoricalDType) variables that should be treated as continuous.
    """
    # Return a modified copy of the input
    data = data.copy()

    treat_cat_as_continuous = [] if treat_cat_as_continuous is None else list(treat_cat_as_continuous)
    treat_cat_as_continuous = [var for var in treat_cat_as_continuous if var in data.columns]
    data[treat_cat_as_continuous] = data[treat_cat_as_continuous].apply(lambda col: col.cat.codes.replace(-1, np.nan))
    return data


# Create ggplot theme
theme_ggplot: p9.theme = p9.theme_minimal()
