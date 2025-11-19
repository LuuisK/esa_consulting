"""
Script for dimensionality reduction.

Run scripts as interactive notebooks in the follwing order:
    preprocessing.py  ->  plot_corr_assoc.py  ->  dimensionality_reduction.py  ->  clustering.py
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotnine as p9
import prince

from utils import check_save_path, load_data, save_data

pd.options.mode.copy_on_write = True

type DimRedMethod = Literal["FAMD", "PCA", "MCA"]
type DimRed = prince.FAMD | prince.PCA | prince.MCA


def reduce_dimen(
    data: pd.DataFrame,
    dim_red_method: DimRedMethod = "PCA",
    n_components: int = 10,
    treat_cat_as_continuous: list[str] | None = None,
    rng: np.random.Generator | np.random.BitGenerator | None = None,
) -> tuple[pd.DataFrame, DimRed, pd.DataFrame]:
    """
    Reduce the dimensionality of the input data to a specified number of coordinates.

    The available methods are PCA, MCA, and FAMD. For ordinal variables to be treated like continuous ones, they need to
    be passed to treat_cat_as_continuous.
    """
    # Create RNG to simulate randomness
    rng = np.random.default_rng(rng)

    # Recode to continuous
    treat_cat_as_continuous = [] if treat_cat_as_continuous is None else treat_cat_as_continuous
    treat_cat_as_continuous = [var for var in treat_cat_as_continuous if var in data.columns]
    data[treat_cat_as_continuous] = data[treat_cat_as_continuous].apply(lambda col: col.cat.codes.replace(-1, np.nan))

    match dim_red_method:
        case "FAMD":
            model = prince.FAMD(
                n_components=n_components,
                n_iter=3,
                random_state=np.random.RandomState(rng.bit_generator),
            )
        case "PCA":
            model = prince.PCA(
                rescale_with_mean=True,
                rescale_with_std=True,
                n_components=n_components,
                n_iter=3,
                random_state=np.random.RandomState(rng.bit_generator),
            )
        case "MCA":
            model = prince.MCA(
                n_components=n_components,
                n_iter=10,
                random_state=np.random.RandomState(rng.bit_generator),
                one_hot=True,
                handle_unknown="error",
            )

    model.fit(data)
    data_loadings = model.row_coordinates(data)
    var_loadings = model.column_coordinates_  # type: ignore

    return data_loadings, model, var_loadings


def plot_reduced_data(
    data_loadings: pd.DataFrame,
    var_loadings: pd.DataFrame | None = None,
    cluster_labels=None,
    comp_x_id: int = 0,
    comp_y_id: int = 1,
    var_marker: str = "x",
) -> p9.ggplot:
    """
    Plots data after dimensionality reduction.

    The columns are the ordered coordinates of reduced dimensionality.
    """
    guide = "legend" if cluster_labels is not None else None
    cluster_labels = len(data_loadings) * [0] if cluster_labels is None else cluster_labels

    # Create data for ggplot
    data_ggplot = data_loadings.rename(lambda x: f"comp_{x}", axis=1).assign(
        **{
            "cluster": pd.Categorical(
                [f"cluster {i}" if i >= 0 else np.nan for i in cluster_labels],
                categories=[f"cluster {i}" for i in sorted(set(cluster_labels) - {-1})],
            )
        }
    )

    # Create plot
    plot = (
        p9.ggplot(data_ggplot, p9.aes(x=f"comp_{comp_x_id}", y=f"comp_{comp_y_id}", fill="cluster"))
        + p9.geom_point(alpha=0.8, stroke=0)
        + p9.scale_fill_brewer(type="qualitative", palette="Set2", na_value="white", guide=guide)
    )
    if var_loadings is not None:
        var_ggplot = (
            None
            if var_loadings is None
            else ((var_loadings * data_loadings.abs().max()).rename(lambda x: f"comp_{x}", axis=1).reset_index())
        )
        if var_marker == "label":
            plot += p9.geom_label(
                p9.aes(x=f"comp_{comp_x_id}", y=f"comp_{comp_y_id}", label="variable", group="variable"),
                size=5,
                data=var_ggplot,
                inherit_aes=False,
                # position=p9.position_jitter()
            )
        else:
            plot += p9.geom_point(data=var_ggplot, fill="green", color="green", alpha=1, shape=var_marker)

    return plot


if __name__ == "__main__":
    # Create RNG for reproducibility
    rng = np.random.default_rng(3295)

    # Hardcoded paths, change to apply to new data
    data_path = Path("data/preprocessed/esa_2018_2021_imputed.pkl")
    vars_info_path = Path("data/preprocessed/esa_vars_info.pkl")

    image_folder = "outputs/dim_red"
    dim_red_plot_save_name_template = "dim_red_{group}.png"
    scree_plot_save_name_template = "scree_{group}.png"

    outputs_folder = "outputs/dim_red"
    model_save_name_template = "esa_2018_2021_model_{group}.pkl"
    data_reduced_save_name_template = "esa_2018_2021_reduced_{group}.pkl"
    merged_data_reduced_save_path = Path(outputs_folder, "esa_2018_2021_reduced.pkl")

    allow_overwrite = True

    # Read in data
    data: pd.DataFrame = load_data(data_path)
    vars_info: pd.DataFrame = load_data(vars_info_path)

    dim_red_methods: dict[str, DimRedMethod] = {
        "socioecon": "FAMD", "alcohol": "PCA", "nicotine": "PCA", "other_drugs": "PCA", "medication": "PCA"
    }  # fmt: skip

    dim_reduced_data: dict[str, pd.DataFrame] = dict()

    for group, dim_red_method in dim_red_methods.items():
        # Restrict to one group
        group_vars = vars_info.loc[vars_info["group"] == group, "name"]
        group_var_types = vars_info.loc[vars_info["group"] == group, "type"]
        group_data = data[group_vars]

        # Conduct dimensionality reduction
        treat_cat_as_continuous: list[str] = vars_info.loc[
            (vars_info["group"] == group) & vars_info["type"].isin(["binary", "ordinal"]), "name"
        ].tolist()
        data_loadings, model, var_loadings = reduce_dimen(
            group_data,
            dim_red_method,
            n_components=4,
            treat_cat_as_continuous=treat_cat_as_continuous,
            rng=rng,
        )

        # Visualize data and variables in coordinates of the first two components
        coordinates_plot = plot_reduced_data(
            data_loadings,
            var_loadings,
            comp_x_id=0,
            comp_y_id=1,
            var_marker="label",
            # var_marker="x",
        )
        # import altair as alt
        # alt.data_transformers.disable_max_rows()
        # coordinates_plot = model.plot(
        #     group_data,
        #     x_component=0,
        #     y_component=1,
        #     show_column_labels=True,
        # )
        coordinates_plot.show()
        dim_red_plot_save_path = Path(image_folder, dim_red_plot_save_name_template.format(group=group))
        check_save_path(dim_red_plot_save_path, allow_overwrite=allow_overwrite)
        coordinates_plot.save(dim_red_plot_save_path)
        print(f"Saved plot of reduced {group} {dim_red_method} data as PNG to {dim_red_plot_save_path}.")

        # Investigate eigenvalues and show scree plot
        print(model.eigenvalues_summary)
        scree_plot = model.scree_plot()
        scree_plot.show()
        scree_plot_save_path = Path(image_folder, scree_plot_save_name_template.format(group=group))
        check_save_path(scree_plot_save_path, allow_overwrite=allow_overwrite)
        scree_plot.save(scree_plot_save_path)
        print(f"Saved {group} {dim_red_method} Scree plot as PNG to {scree_plot_save_path}.")

        # Save fitted model and data with reduced dimensionality (factor loadings of data)
        model_save_path = Path(outputs_folder, model_save_name_template.format(group=group))
        save_data(model, model_save_path, label=f"fitted {group} {dim_red_method} model")

        data_reduced_save_path = Path(outputs_folder, data_reduced_save_name_template.format(group=group))
        save_data(
            data_loadings,
            data_reduced_save_path,
            label=f"{group} data with reduced dimensionality ({dim_red_method} factor loadings of data)",
        )

        dim_reduced_data[group] = data_loadings.rename(columns=lambda x: f"{group}_comp_{x}")

    # Merge data from groups
    merged_data_reduced = pd.concat(dim_reduced_data.values(), axis=1)
    # import itertools
    # non_reduced_vars = vars_info.loc[~ vars_info["group"].isin(dim_reduced_data.keys()), "name"]
    # merged_data_reduced = pd.concat(itertools.chain((data[non_reduced_vars],), dim_reduced_data.values()), axis=1)
    save_data(
        merged_data_reduced,
        merged_data_reduced_save_path,
        label="merged data from all groups with reduced dimensionality",
    )
