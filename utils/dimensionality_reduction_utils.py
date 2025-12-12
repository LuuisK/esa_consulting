"""Module with functions for dimensionality_reduction.ipynb"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotnine as p9
from prince import FAMD, MCA, PCA

from utils.utils import theme_ggplot, turn_cats_to_continuous

pd.options.mode.copy_on_write = True

DimRedMethod = FAMD | PCA | MCA
dim_red_method_list = [FAMD, PCA, MCA]
str_to_dim_red_method: dict[str, type[DimRedMethod]] = {x.__name__.lower(): x for x in dim_red_method_list}


def reduce_dimen(
    data: pd.DataFrame,
    dim_red_method: str | type[DimRedMethod] = "PCA",
    n_components: int = 10,
    treat_cat_as_continuous: list[str] | None = None,
    rng: np.random.Generator | np.random.BitGenerator | None = None,
) -> tuple[pd.DataFrame, DimRedMethod, pd.DataFrame]:
    """
    Reduce the dimensionality of the input data to a specified number of coordinates.

    The available methods are PCA, MCA, and FAMD. For categorical (pandas.CategoricalDType) variables to be treated like
    continuous ones, they need to be passed to treat_cat_as_continuous.
    """
    # Create RNG to simulate randomness
    rng = np.random.default_rng(rng)

    # Recode to continuous
    data = turn_cats_to_continuous(data, treat_cat_as_continuous)

    if isinstance(dim_red_method, str):
        # Get model from string
        model_type = str_to_dim_red_method[dim_red_method.lower()]
    elif issubclass(dim_red_method, DimRedMethod):
        model_type = dim_red_method

    # Initialize model
    random_state = np.random.RandomState(rng.bit_generator)
    if issubclass(model_type, FAMD):
        model = model_type(n_components=n_components, n_iter=3, random_state=random_state)
    elif issubclass(model_type, PCA):
        model = model_type(n_components=n_components, n_iter=3, random_state=random_state)
    elif issubclass(model_type, MCA):
        model = model_type(
            n_components=n_components, n_iter=10, random_state=random_state, one_hot=True, handle_unknown="error"
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
        + theme_ggplot
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
