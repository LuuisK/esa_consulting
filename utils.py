import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import Axes, Figure


def check_save_path(save_path: str, allow_overwrite: bool = True):
    """
    Check that no file would be overwritten, and create the folder to save into if it doesn't exist yet.
    """
    save_folder, filename = os.path.split(save_path)
    if os.path.exists(save_path) and not allow_overwrite:
        raise FileExistsError(f"File {save_path} already exists and would be overwritten.")
    os.makedirs(save_folder, exist_ok=True)


def plot_corr(data: pd.DataFrame, var_names: list[str], **kwargs) -> tuple[Figure, Axes]: # TODO: fix to work with cats
    """
    Plot the correlation matrix of the variables in the given list.
    """
    tick_label_size = kwargs.get("tick_label_size", 11)
    title_size = kwargs.get("title_size", 16)
    title = kwargs.get("title", "Correlation Matrix")
    figsize = kwargs.get("figsize", (10, 8))

    data = data.loc[:,(*var_names,)]
    data = data.apply(lambda x: pd.Categorical(x).codes, axis=0)
    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=figsize)
    corr_mat = ax.matshow(data.corr(), cmap="PuOr_r", vmin=-1, vmax=1)  # "coolwarm", "RdBu_r", "BrBG_r", "PuOr_r"
    plt.xticks(
        range(data.select_dtypes(["number"]).shape[1]), data.select_dtypes(["number"]).columns, fontsize=tick_label_size, rotation=90
    )
    plt.yticks(range(data.select_dtypes(["number"]).shape[1]), data.select_dtypes(["number"]).columns, fontsize=tick_label_size)
    cb = fig.colorbar(corr_mat)
    cb.ax.tick_params(labelsize=14)
    ax.set_title(title, fontsize=title_size)
    return fig, ax
