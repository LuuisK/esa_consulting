"""Module with utilities for plotting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotnine as p9
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.mixture._base import BaseMixture
from sklearn.model_selection import GridSearchCV

from utils.corr_assoc_utils import create_corr_assoc_matrix
from utils.misc_utils import contingency_table

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


# Create ggplot theme
theme_ggplot: p9.theme = p9.theme_minimal()
default_color_ggplot: str = "midnightblue"


def make_plot_2d_data(data: pd.DataFrame, cluster_labels: pd.Series[pd.CategoricalDtype] | None = None):
    """
    Creates a plot of a 2d dataset.

    The x and y coordinates are the first and second column of the data. If cluster labels are provided the points are
    colored by cluster membership.
    """
    # Copy data so the original object is not modified
    data = data.copy()

    # Color based on clustering if provided
    aes_kwargs = {}
    geom_point_kwargs = {}
    if cluster_labels is None:
        geom_point_kwargs |= dict(fill=default_color_ggplot)
    else:
        data["cluster"] = cluster_labels
        aes_kwargs |= dict(fill="cluster")

    # Create plot
    plot = (
        p9.ggplot(data)
        + p9.aes(x=data.columns[0], y=data.columns[1], **aes_kwargs)
        + p9.geom_point(alpha=0.3, **geom_point_kwargs, color="none")
        + theme_ggplot
    )
    return plot


def make_plot_corr_assoc_matrix(data: pd.DataFrame) -> p9.ggplot:
    """
    Creates a matrix containing values that measure the correlation or association between variables.

    See create_corr_assoc_matrix for the correlation and association measures used.
    """
    # Create correlation matrix
    corr_assoc_matrix, p_value_matrix = create_corr_assoc_matrix(data)

    # Order variables into blocks
    categorical_vars = [var for var in data.columns if isinstance(data[var].dtype, pd.CategoricalDtype)]
    continuous_vars = set(data.columns) - set(categorical_vars)
    vars_ordered = [*continuous_vars, *categorical_vars]

    # Pivot matrices
    corr_assocs = corr_assoc_matrix.stack(future_stack=True).rename("corr_assoc")  # type: ignore
    p_values = p_value_matrix.stack(future_stack=True).rename("p_value")  # type: ignore

    # Create dataset
    def _get_type(values):
        return "qualitative" if isinstance(values.dtype, pd.CategoricalDtype) else "quantitative"

    data_ggplot = (
        pd.merge(corr_assocs, p_values, left_index=True, right_index=True)
        .reset_index(names=["vari", "varj"])
        .astype(
            {
                "vari": pd.CategoricalDtype(categories=vars_ordered),
                "varj": pd.CategoricalDtype(categories=vars_ordered[::-1]),
                "corr_assoc": np.float64,
                "p_value": np.float64,
            }
        )
        .assign(
            **{
                "i_type": lambda x: pd.Categorical(
                    x.apply(lambda row: _get_type(data[row["vari"]]), axis=1), ["quantitative", "qualitative"]
                ),
                "j_type": lambda x: pd.Categorical(
                    x.apply(lambda row: _get_type(data[row["varj"]]), axis=1), ["quantitative", "qualitative"]
                ),
                "significant": lambda x: pd.Categorical((x["p_value"] < 0.05).astype(str), categories=["False"]),
            }
        )
    )

    # Fix plot labels
    var_scale = data_ggplot["i_type"].unique()
    if "qualitative" not in var_scale:
        corr_assoc_str = "correlation"
        n_types = 1
    elif "quantitative" not in var_scale:
        corr_assoc_str = "association"
        n_types = 1
    else:
        corr_assoc_str = "correlation /\nassociation"
        n_types = 2

    # Create plot
    corr_assoc_plot = (
        p9.ggplot(data_ggplot, p9.aes(x="vari", y="varj", fill="corr_assoc"))
        + p9.geom_tile(width=1)
        + (p9.facet_grid(rows="j_type", cols="i_type", scales="free", space="free") if n_types == 2 else None)
        + p9.labs(x="", y="", fill=corr_assoc_str)
        + p9.scale_fill_gradient(low="midnightblue", high="white", na_value="red", limits=(0, 1))
        + p9.coord_fixed()
        + p9.geom_point(p9.aes(x="vari", y="varj", shape="significant"), inherit_aes=False)
        + p9.scale_shape_manual(
            values={"False": "x"}, name=" ", labels={"False": f"no significant\n{corr_assoc_str}"}, na_translate=False
        )
        + theme_ggplot
        + p9.theme(axis_text_x=p9.element_text(rotation="vertical"))
    )
    return corr_assoc_plot


def make_plot_scree(pca: PCA) -> p9.ggplot:
    """
    Creates the scree plot of a PCA model.
    """
    exp_var_rat = pca.explained_variance_ratio_
    data_ggplot = pd.DataFrame({"pc": np.arange(len(exp_var_rat)), "exp_var_rat": exp_var_rat}).assign(
        **{
            "cum_exp_var_rat": lambda x: np.cumsum(x["exp_var_rat"]),
            "bar": lambda x: pd.Categorical(["" for _ in x.index], categories=[""]),
            "line": lambda x: pd.Categorical(["" for _ in x.index], categories=[""]),
        }
    )

    plot_scree = (
        p9.ggplot(data_ggplot)
        + p9.aes(x="pc")
        + p9.geom_bar(p9.aes(y="exp_var_rat", fill="bar"), stat=p9.stat_identity())
        + p9.geom_line(p9.aes(y="cum_exp_var_rat", color="line"))
        + p9.labs(x="Principal component", y="Explained variance ratio")
        + p9.scale_fill_manual(values={"": "steelblue"}, name=" ", labels={"": "Explained variance\nratio"})
        + p9.scale_color_manual(values={"": "black"}, name=" ", labels={"": "Cumulative explained\nvariance ratio"})
        + theme_ggplot
        + p9.theme(legend_position="bottom", legend_direction="horizontal")
    )
    return plot_scree


def make_plot_cv(cv_model: GridSearchCV, by: str | None = None, group: str | None = None) -> p9.ggplot:
    """
    Creates a plot of cross validation performances for different parameters.
    """
    by = next(iter(cv_model.param_grid[0])) if by is None else by  # type: ignore

    data_ggplot = pd.DataFrame(cv_model.cv_results_["params"]).assign(
        **{
            "is_best": lambda x: x.apply(lambda y: y.equals(pd.Series(cv_model.best_params_)), axis=1),
            "mean_test_score": cv_model.cv_results_["mean_test_score"],
        }
    )

    plot_cv = (
        p9.ggplot(data_ggplot)
        + (p9.aes(by, "- mean_test_score", color=group) if group is not None else p9.aes(by, "- mean_test_score"))
        + p9.geom_line()
        + p9.geom_point(
            p9.aes(by, "- mean_test_score"),
            inherit_aes=False,
            data=data_ggplot[data_ggplot["is_best"]],
            shape="o",
            size=5,
            color="purple",
            fill="none",
        )
        + p9.geom_vline(
            p9.aes(xintercept=by),
            inherit_aes=False,
            data=data_ggplot[data_ggplot["is_best"]],
            color="purple",
            linetype="--",
            alpha=0.5,
        )
        + theme_ggplot
    )
    return plot_cv


def make_plot_mixture_weights(model: BaseMixture) -> p9.ggplot:
    """
    Creates a plot of mixture weights.
    """
    model_weights = (
        pd.DataFrame({"weight": model.weights_})  # type: ignore
        .sort_values("weight")
        .assign(**{"sorted_component": lambda x: range(len(x))})
    )

    plot_weights = (
        p9.ggplot(model_weights)
        + p9.aes("sorted_component", "weight")
        + p9.geom_line()
        + p9.scale_y_log10()
        + p9.aes(ymax=1.0)
        + theme_ggplot
    )
    return plot_weights


def make_plot_silhouette(data: pd.DataFrame, cluster_labels: pd.Series[pd.CategoricalDtype]) -> p9.ggplot:
    """Creates a silhouette plot."""
    # Compute silhouettes
    silhouettes = silhouette_samples(data, cluster_labels, metric="euclidean")

    # Create ggplot data
    data_ggplot = pd.DataFrame({"cluster": cluster_labels, "silhouette": silhouettes})
    data_ggplot["rank"] = (
        data_ggplot.groupby("cluster")["silhouette"]
        .rank(method="first", ascending=False)
        .astype("int")
        .astype("category")
    )

    # Create silhouette plot
    silhouette_plot = (
        p9.ggplot(data_ggplot)
        + p9.aes("rank", "silhouette", group="cluster", color="cluster", fill="cluster")
        + p9.facet_grid(rows=None, cols="cluster", scales="free_x", space="free_x")
        + p9.geom_bar(stat="identity", width=0)
        + p9.labs(x="")
        + theme_ggplot
        + p9.theme(
            axis_text_x=p9.element_blank(),
            axis_ticks_x=p9.element_blank(),
            panel_grid=p9.element_blank(),
            strip_text=p9.element_blank(),
            panel_grid_major_y=p9.element_line(color="gray", alpha=0.5),
            panel_spacing=0,
            # panel_border=p9.element_rect(color="gray", alpha=0.5),
            plot_background=p9.element_rect(fill="gray", alpha=0.3),
        )
    )
    return silhouette_plot


def make_plot_contingency_table(
    values1: pd.Series[pd.CategoricalDtype],
    values2: pd.Series[pd.CategoricalDtype],
    name1: str = "var1",
    name2: str = "var2",
) -> p9.ggplot:
    """
    Creates a plot of the contingency table between two categorical variables.

    The plot shows how well the categories of the categorical variable values2 correspond to each category of the
    categorical variable values1, in percentages for each value of values1. The indices of values1 and values2 should
    match.
    """
    cont_table = contingency_table(values1, values2, dropna=False, normalize="index")
    data_ggplot = pd.DataFrame({"entailment": cont_table.stack(future_stack=True)}).reset_index(names=[name1, name2])

    cont_table_plot = (
        p9.ggplot(data_ggplot)
        + p9.aes(x=name2, y=name1, fill="entailment")
        + p9.geom_tile(height=0.975, width=0.975)
        + theme_ggplot
    )
    return cont_table_plot
