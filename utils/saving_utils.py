"""Module with utilities for saving."""

from __future__ import annotations

import os
import pickle
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotnine as p9
from plotnine.composition import Compose
from plotnine.exceptions import PlotnineWarning

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


@dataclass
class DatasetBundle:
    """
    Dataset bundle.

    Bundle of a dataset and optionally metadata about variables and missingness information.
    """

    data: pd.DataFrame
    meta: pd.DataFrame | None = None
    na: pd.DataFrame | None = None

    def save(self, path: Path) -> None:
        """
        Save to parquet file(s).

        The path may not have an extension. If they exist, metadata about variables and missingness info are saved as
        well.
        """
        self.data.to_parquet(f"{path}.parquet")
        if self.meta is not None:
            self.meta.to_parquet(f"{path}__meta.parquet", index=False)
        if self.na is not None:
            self.na.to_parquet(f"{path}__na.parquet", index=False)
        print(f"Saved dataset at {path}.")

    @classmethod
    def load(cls, path: str | Path) -> DatasetBundle:
        """
        Load from parquet file(s).

        The path may not have an extension. If they exist, metadata about variables and missingness info are loaded as
        well.
        """
        path = Path(path)
        if path.suffix != "":
            raise ValueError("'path' may not have an extension.")
        folder = path.parent
        data_name = path.name
        files = os.listdir(folder)
        return cls(
            pd.read_parquet(Path(folder, f"{data_name}.parquet")),
            pd.read_parquet(Path(folder, file_name)) if (file_name := f"{data_name}__meta.parquet") in files else None,
            pd.read_parquet(Path(folder, file_name)) if (file_name := f"{data_name}__na.parquet") in files else None,
        )
        print(f"Loaded dataset from {path}.")

    def copy(self) -> DatasetBundle:
        """Copy."""
        return DatasetBundle(
            self.data.copy(),
            None if self.meta is None else self.meta.copy(),
            None if self.na is None else self.na.copy(),
        )

    def drop(self, vars: str | Iterable[str]) -> DatasetBundle:
        vars = [vars] if isinstance(vars, str) else list(vars)
        return DatasetBundle(
            self.data.drop(columns=vars),
            None if self.meta is None else self.meta[lambda x: ~x["name"].isin(vars)],
            None if self.na is None else self.na.drop(columns=vars),
        )


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


def save_object(data: Any, save_path: str | Path) -> None:
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


def load_object(file_path: str | Path) -> Any:
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
