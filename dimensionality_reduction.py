"""
Script for dimensionality reduction.

Run scripts as interactive notebooks in the follwing order:
    preprocessing.py  ->  plot_corr_assoc.py  ->  dimensionality_reduction.py  ->  clustering.py
"""

import os
import pickle

import altair as alt
import numpy as np
import pandas as pd
import prince

from utils import check_save_path

pd.options.mode.copy_on_write = True
alt.data_transformers.disable_max_rows()


# Hardcoded paths, change to apply to new data
data_folder = os.path.join("data", "preprocessed")
data_path = os.path.join(data_folder, "esa_2018_2021_imputed.pkl")
vars_info_path = os.path.join(data_folder, "esa_vars_info.pkl")

image_folder = os.path.join("outputs", "images")
famd_plot_save_path_template = os.path.join(image_folder, "famd_{}.png")
scree_plot_save_path_template = os.path.join(image_folder, "scree_{}.png")

outputs_folder = "outputs"
famd_save_path_template = os.path.join(outputs_folder, "esa_2018_2021_famd_{}.pkl")
data_reduced_save_path_template = os.path.join(outputs_folder, "esa_2018_2021_reduced_{}.pkl")

allow_overwrite = True


if __name__ == "__main__":
    # Create RNG for reproducibility
    rng = np.random.default_rng(3295)

    # Read in data
    data: pd.DataFrame = pd.read_pickle(data_path)
    vars_info: pd.DataFrame = pd.read_pickle(vars_info_path)

    # Restrict to one group
    group = "alcohol"
    group_vars = vars_info.loc[vars_info["group"] == group, "variable"]
    data = data[group_vars]

    # Conduct a factor analysis of mixed data for dimensionality reduction
    famd = prince.FAMD(
        n_components=10,
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=np.random.RandomState(rng.bit_generator),
    )
    famd.fit(data)
    data_factor_loadings = famd.row_coordinates(data)
    var_factor_loadings = famd.column_coordinates_

    # Visualize data and variables in coordinates of first two components
    famd_plot = famd.plot(
        data,
        x_component=0,
        y_component=1,
        show_column_labels=True,
    )
    famd_plot.show()
    famd_plot_save_path = famd_plot_save_path_template.format(group)
    check_save_path(famd_plot_save_path, allow_overwrite=allow_overwrite)
    famd_plot.save(famd_plot_save_path)
    print(f"Saved FAMD plot as PNG to {famd_plot_save_path}.")

    # Investigate eigenvalues and show scree plot
    print(famd.eigenvalues_summary)
    scree_plot = famd.scree_plot()
    scree_plot.show()
    scree_plot_save_path = scree_plot_save_path_template.format(group)
    check_save_path(scree_plot_save_path, allow_overwrite=allow_overwrite)
    scree_plot.save(scree_plot_save_path)
    print(f"Saved Scree plot as PNG to {scree_plot_save_path}.")

    # Save FAMD output and data with reduced dimensionality (factor loadings of data)
    famd_save_path = famd_save_path_template.format(group)
    check_save_path(famd_save_path, allow_overwrite=allow_overwrite)
    with open(famd_save_path, "wb") as file:
        pickle.dump(famd, file, protocol=-1)
    print(f"Saved FAMD output as Pickle to {famd_save_path}.")

    data_reduced_save_path = data_reduced_save_path_template.format(group)
    check_save_path(data_reduced_save_path, allow_overwrite=allow_overwrite)
    data_factor_loadings.to_pickle(data_reduced_save_path)
    print(f"Saved data with reduced dimensionality (factor loadings of data) as Pickle to {data_reduced_save_path}.")
