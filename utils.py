"""
Script for defining helpful functions.
"""

import os


def check_save_path(save_path: str, allow_overwrite: bool = True):
    """
    Controls a save path.

    Checks that no file would be overwritten, and creates the folder to save into if it doesn't exist yet.
    """
    save_folder, filename = os.path.split(save_path)
    if os.path.exists(save_path) and not allow_overwrite:
        raise FileExistsError(f"File {save_path} already exists and would be overwritten.")
    os.makedirs(save_folder, exist_ok=True)
