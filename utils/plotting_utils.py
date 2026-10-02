"""Module with utilities for plotting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotnine as p9
from matplotlib import font_manager
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA
from sklearn.mixture._base import BaseMixture
from sklearn.model_selection import GridSearchCV

from utils.bootstrap_utils import BootstrapGridSearch
from utils.corr_assoc_utils import create_corr_assoc_matrix
from utils.metrics_utils import silhouette_samples
from utils.misc_utils import (
    agg_with_overall,
    clip_data_quantiles,
    crosstab,
    partial_fn_with_info,
    std_weighted,
    turn_cats_to_continuous,
)

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


# Register font
font_paths = [
    Path("fonts/Libertinus-7.051/LibertinusSerif-Regular.otf"),
    Path("fonts/Libertinus-7.051/LibertinusSerif-Bold.otf"),
    Path("fonts/Libertinus-7.051/LibertinusSerif-Italic.otf"),
    Path("fonts/Libertinus-7.051/LibertinusSerif-BoldItalic.otf"),
]
for font_path in font_paths:
    font_manager.fontManager.addfont(font_path)
font_name = font_manager.FontProperties(fname=font_paths[0]).get_name()


# Create ggplot theme
default_color = "black"

colors_pyplot = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

theme_ggplot: p9.theme = p9.theme_bw(base_family=font_name) + p9.theme(
    text=p9.element_text(color=default_color),
    axis_text=p9.element_text(color=default_color),
    axis_title=p9.element_text(color=default_color),
    legend_text=p9.element_text(color=default_color),
    legend_title=p9.element_text(color=default_color),
    plot_title=p9.element_text(color=default_color),
    strip_text=p9.element_text(color=default_color),
)

scale_color_ggplot = p9.scale_color_hue()
scale_fill_ggplot = p9.scale_fill_hue()


def make_plot_2d_data(
    data: pd.DataFrame,
    comps: tuple[int, int] | list[tuple[int, int]],
    labels: pd.Series | None = None,
    quantiles: tuple[float, float] = (0, 1),
    method: Literal["clip", "drop"] = "clip",
) -> p9.ggplot:
    """
    Creates a plot of a 2D dataset.

    Which columns are plotted as x and y coordinates can be specified by position in `comps`. If cluster labels are
    provided the points are colored by cluster membership. The `quantiles` and `method` parameters are used to clip
    extreme quantiles in each facet of the plot.
    """

    # Copy data so the original object is not modified
    data = data.copy()

    single_plot = isinstance(comps, tuple)
    comps = [comps] if single_plot else comps

    # Data-frame of every combination of components with clipped extreme percentiles
    dims_dtype = pd.CategoricalDtype([f"{data.columns[i]} vs {data.columns[j]}" for i, j in comps])
    data_2d = pd.concat(
        [
            pd.DataFrame(
                {
                    "x": data.iloc[:, i],
                    "y": data.iloc[:, j],
                    "dims": pd.Series(dims_dtype.categories[k], index=data.index, dtype=dims_dtype),
                }
                | ({} if labels is None else {"cluster": labels})
            )
            .pipe(clip_data_quantiles, var_name="x", quantiles=quantiles, method=method)
            .pipe(clip_data_quantiles, var_name="y", quantiles=quantiles, method=method)
            for k, (i, j) in enumerate(comps)
        ],
        ignore_index=True,
    )

    # Color based on clustering if provided
    aes_kwargs = {}
    if labels is not None:
        aes_kwargs |= {"fill": "cluster"}

    # Create plot
    data_2d = data_2d.rename(columns=lambda x: str(x))
    plot = (
        p9.ggplot(data_2d)
        + p9.aes(x=data_2d.columns[0], y=data_2d.columns[1], **aes_kwargs)
        + p9.geom_point(alpha=0.3, color="none")
        + scale_fill_ggplot
        + p9.guides(fill=p9.guide_legend(override_aes={"size": 5, "alpha": 1}))
        + theme_ggplot
    )
    if single_plot:
        return plot + p9.labs(x=data.columns[comps[0][0]], y=data.columns[comps[0][1]])
    else:
        return (
            plot
            + p9.theme(axis_title_x=p9.element_blank(), axis_title_y=p9.element_blank())
            + p9.facet_wrap("dims", scales="free")
        )


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
    corr_assocs = corr_assoc_matrix.stack(future_stack=True).rename("corr_assoc")
    p_values = p_value_matrix.stack(future_stack=True).rename("p_value")

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
    by = next(iter(cv_model.param_grid[0])) if by is None else by

    data_ggplot = pd.DataFrame(cv_model.cv_results_["params"]).assign(
        **{
            "is_best": lambda x: x.apply(lambda y: y.equals(pd.Series(cv_model.best_params_)), axis=1),
            "mean_score": cv_model.cv_results_["mean_test_score"],
        }
    )
    data_best = data_ggplot[lambda x: x["is_best"]]

    plot_cv = (
        p9.ggplot(data_ggplot)
        + (p9.aes(by, "- mean_score", color=group) if group is not None else p9.aes(by, "- mean_score"))
        + p9.geom_line()
        + p9.geom_point(
            p9.aes(by, "- mean_score"), data=data_best, inherit_aes=False, shape="o", size=5, color="black", fill="none"
        )
        + p9.geom_vline(
            p9.aes(xintercept=by), data=data_best, inherit_aes=False, color="black", linetype="--", alpha=0.5
        )
        + p9.scale_x_continuous(expand=(0, 0))
        + scale_color_ggplot
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


def make_plot_silhouette(
    data: pd.DataFrame, cluster_labels: pd.Series[pd.CategoricalDtype], sample_weight: ArrayLike | None = None
) -> p9.ggplot:
    """Creates a silhouette plot."""
    sample_weight = None if sample_weight is None else pd.Series(sample_weight, index=data.index)

    # Compute silhouettes
    silhouettes = silhouette_samples(data, cluster_labels, sample_weight=sample_weight, metric="euclidean")

    # Create ggplot data
    data_ggplot = pd.DataFrame({"cluster": cluster_labels, "silhouette": silhouettes})
    data_ggplot["rank_group"] = (
        data_ggplot.groupby("cluster")["silhouette"].rank(method="first", ascending=False).astype("int")
    )
    data_ggplot = data_ggplot.sort_values(["cluster", "rank_group"])
    data_ggplot["rank"] = range(len(data_ggplot))

    # Create silhouette plot
    silhouette_plot = (
        p9.ggplot(data_ggplot)
        + p9.aes(x="rank", y="silhouette", color="cluster", fill="cluster")
        + p9.geom_bar(stat="identity", width=0)
        + p9.labs(x="")
        + p9.lims(y=(-1, 1))
        + p9.scale_x_continuous(expand=(0, 0))
        + scale_color_ggplot
        + scale_fill_ggplot
        + theme_ggplot
        + p9.theme(
            panel_grid_major_x=p9.element_blank(),
            panel_grid_minor_x=p9.element_blank(),
            axis_text_x=p9.element_blank(),
            axis_ticks_x=p9.element_blank(),
        )
    )
    return silhouette_plot


def make_plot_contingency_table(
    values1: pd.Series[pd.CategoricalDtype],
    values2: pd.Series[pd.CategoricalDtype],
    name1: str = "var1",
    name2: str = "var2",
    sample_weight: ArrayLike | None = None,
) -> p9.ggplot:
    """
    Creates a plot of the contingency table between two categorical variables.

    The plot shows how well the categories of the categorical variable values2 correspond to each category of the
    categorical variable values1, in percentages for each value of values1. The indices of values1 and values2 should
    match.
    """
    cont_table = crosstab(values1, values2, sample_weight=sample_weight, dropna=False, normalize="index")
    data_ggplot = (
        pd.DataFrame({"entailment": cont_table.stack(future_stack=False)})
        .reset_index(names=[name1, name2])
        .assign(
            **{
                "text": lambda x: x["entailment"].apply(lambda y: f"{y:.2f}"),
                "text_color": lambda x: pd.cut(x["text"].astype("float"), (0, 0.4, 1), labels=("low", "high")),
            }
        )
    )

    cont_table_plot = (
        p9.ggplot(data_ggplot)
        + p9.aes(x=name2, y=name1, fill="entailment")
        + p9.geom_tile(height=0.975, width=0.975)
        + p9.geom_text(p9.aes(label="text", color="text_color"), size=9, show_legend=False)
        + p9.scale_color_manual(["white", "black"], na_value="gray")  # na_value for 0.00
        + p9.coord_fixed()
        + theme_ggplot
    )
    return cont_table_plot


def make_plot_grid_search(
    metrics_grid_search: pd.DataFrame, x: str, metrics: str | tuple[str] | None = None
) -> p9.ggplot:
    """
    Creates a plot of grid-search metrics.

    If metrics is `None`, all metrics are plotted. The line and error bars show the estimated mean with its Monte Carlo
    standard error, the ribbon shows one standard deviation across bootstrapping runs around the mean.
    """
    if isinstance(metrics, str):
        metrics = (metrics,)
    elif metrics is None:
        metrics = tuple(metrics_grid_search["metric"].unique())

    data_ggplot = metrics_grid_search[lambda x: x["metric"].isin(metrics)]
    return (
        p9.ggplot(data_ggplot)
        + p9.aes(x=x, y="mean")
        + p9.geom_ribbon(p9.aes(ymin="mean - std", ymax="mean + std"), alpha=0.15)
        + p9.geom_line()
        + p9.geom_errorbar(p9.aes(ymin="mean - mcse", ymax="mean + mcse"), width=0.15)
        + (None if len(metrics) <= 1 else p9.facet_wrap("metric", scales="free_y"))
        + p9.labs(x=x, y=metrics[0] if len(metrics) == 1 else "values")
        + theme_ggplot
    )


def make_plot_parameter_selection(
    grid_search_boot: BootstrapGridSearch, x_param: str
) -> tuple[p9.ggplot, p9.ggplot, p9.ggplot | None]:
    """
    Plot the two-stage parameter selection.

    Returns:
    - A plot of the selection metric.
    - A plot of the differences in the selection metric between different parameter combinations.
    - If it is a metric, a plot of the tie breaking metric, otherwise `None`.

    The eligible parameter combinations are colored. The parameter combination that maximizes the selection metric and
    the selected parameter combination are circled.
    """
    colors = ["green", "black", "darkblue", "orange"]
    x_params_eligible = [params[x_param] for params in grid_search_boot.eligible_params_]

    def eligible_fn(x: pd.DataFrame):
        return x[x_param].isin(x_params_eligible).map({True: "eligible", False: "uneligible"})

    summaries_with_eligibility = grid_search_boot.summaries_.assign(**{"eligible": eligible_fn})
    x_param_selected = grid_search_boot.best_params_[x_param]

    # Selection metric plot
    selection_metric_summaries = summaries_with_eligibility[lambda x: x["metric"].eq(grid_search_boot.selection_metric)]
    x_param_selection_metric_max = selection_metric_summaries.loc[lambda x: x["mean"].idxmax(), x_param]
    selection_metric_points = selection_metric_summaries[
        lambda x: x[x_param].isin([x_param_selection_metric_max, x_param_selected])
    ].assign(
        **{
            "point_type": lambda x: x[x_param].map(
                {x_param_selection_metric_max: "maximum", x_param_selected: "selected"}
            )
        }
    )
    selection_metric_colors = {
        k: v
        for k, v in zip(["eligible", "uneligible", "maximum", "selected"], colors, strict=True)
        if k in {*selection_metric_summaries["eligible"], *selection_metric_points["point_type"]}
    }
    selection_metric_plot = (
        p9.ggplot(selection_metric_summaries)
        + p9.aes(x=x_param, y="mean", color="eligible", fill="eligible", group=0)
        # Line
        + p9.geom_line()
        # Error bars and ribbon
        + p9.geom_errorbar(p9.aes(ymin="mean - mcse", ymax="mean + mcse"), width=1e-10)
        + p9.geom_ribbon(p9.aes(x=x_param, ymin="mean - std", ymax="mean + std"), inherit_aes=False, alpha=0.15)
        # Point
        + p9.geom_point(p9.aes(color="point_type"), data=selection_metric_points, shape="o", fill="none", size=3)
        # Labels, scales, legend guides, theme
        + p9.labs(y=grid_search_boot.selection_metric)
        + p9.scale_color_manual(selection_metric_colors, breaks=list(selection_metric_colors))
        + theme_ggplot
        + p9.guides(
            color=p9.guide_legend(
                "colors", override_aes={"linetype": "", "shape": "s", "fill": list(selection_metric_colors.values())}
            )
        )
    )

    # Differences plot
    difference_summaries = grid_search_boot.paired_difference_summaries_.assign(
        **{"eligible": eligible_fn, "linetype1": "mean", "linetype2": "lower bound", "linetype3": "zero"}
    )
    differences_points = difference_summaries[
        lambda x: x[x_param].isin([x_param_selection_metric_max, x_param_selected])
    ].assign(
        **{
            "point_type": lambda x: x[x_param].map(
                {
                    x_param_selection_metric_max: f"maximum {grid_search_boot.selection_metric}",
                    x_param_selected: "selected",
                }
            )
        }
    )
    differences_colors = {
        k: v
        for k, v in zip(
            ["eligible", "uneligible", f"maximum {grid_search_boot.selection_metric}", "selected"], colors, strict=True
        )
        if k in {*difference_summaries["eligible"], *differences_points["point_type"]}
    }
    differences_plot = (
        p9.ggplot(difference_summaries)
        + p9.aes(x=x_param, y="mean", color="eligible", fill="eligible", group=0)
        # Lines
        + p9.geom_line(p9.aes(linetype="linetype1"))
        + p9.geom_line(p9.aes(y="lower_bound", linetype="linetype2"))
        + p9.geom_hline(p9.aes(yintercept=0, linetype="linetype3"))
        # Error bars and ribbon
        + p9.geom_errorbar(p9.aes(ymin="lower_bound", ymax="mean"), width=1e-10, alpha=0.5)
        + p9.geom_ribbon(p9.aes(x=x_param, ymin="lower_bound", ymax="mean"), inherit_aes=False, alpha=0.15)
        # Points
        + p9.geom_point(p9.aes(color="point_type"), data=differences_points, shape="o", fill="none", size=3)
        # Labels, scales, legend guides, theme
        + p9.labs(y="difference to maximum")
        + p9.scale_color_manual(differences_colors, breaks=list(differences_colors))
        + p9.scale_linetype_manual({"mean": "solid", "lower bound": "dashed", "zero": "dotted"})
        + p9.guides(
            color=p9.guide_legend(
                "colors", override_aes={"linetype": "", "shape": "s", "fill": list(differences_colors.values())}
            ),
            linetype=p9.guide_legend(title="lines", position="right"),
        )
        + theme_ggplot
    )

    # Tie break plot
    if grid_search_boot.tie_break is None:
        tie_break_plot = None
    else:
        tie_break_summaries = summaries_with_eligibility[lambda x: x["metric"].eq(grid_search_boot.tie_break)]
        tie_break_points = tie_break_summaries[lambda x: x[x_param].eq(x_param_selected)].assign(
            **{"point_type": "selected"}
        )
        tie_break_colors = {
            k: v
            for k, v in zip(["eligible", "uneligible", "selected"], [colors[i] for i in (0, 1, 3)], strict=True)
            if k in {*tie_break_summaries["eligible"], *tie_break_points["point_type"]}
        }
        tie_break_plot = (
            p9.ggplot(tie_break_summaries)
            + p9.aes(x=x_param, y="mean", color="eligible", fill="eligible", group=0)
            # Line
            + p9.geom_line()
            # Error bars and ribbon
            + p9.geom_errorbar(p9.aes(ymin="mean - mcse", ymax="mean + mcse"), width=1e-10)
            + p9.geom_ribbon(p9.aes(x=x_param, ymin="mean - std", ymax="mean + std"), inherit_aes=False, alpha=0.15)
            # Point
            + p9.geom_point(p9.aes(color="point_type"), data=tie_break_points, shape="o", fill="none", size=3)
            # Labels, scales, legend guides, theme
            + p9.labs(y=grid_search_boot.tie_break)
            + p9.scale_color_manual(tie_break_colors, breaks=list(tie_break_colors))
            + theme_ggplot
            + p9.guides(
                color=p9.guide_legend(
                    "colors", override_aes={"linetype": "", "shape": "s", "fill": list(tie_break_colors.values())}
                )
            )
        )

    return selection_metric_plot, differences_plot, tie_break_plot


def make_plot_continuous_var_dist_by_cluster(
    series: ArrayLike,
    labels: ArrayLike,
    quantiles: Iterable[float] | None = None,
    sample_weight: ArrayLike | None = None,
    sort: bool = False,
    ascending: bool = False,
    cluster_size_decimals: int = 0,
    column_size: int = 4,
) -> p9.ggplot:
    """
    Creates the plot of the distribution of a continuous variable in each cluster.

    If a categorical variable is given, its (ordered) category codes are treated as a continuous variable. Above each
    distribution, the (effective) cluster size is shown. If sort the clusters in the plot are sorted by mean.
    """
    series = pd.Series(series)

    quantiles = [0.02, 0.1, 0.33, 0.5, 0.67, 0.9, 0.98] if quantiles is None else list(quantiles)
    if 0.5 not in quantiles:
        quantiles = [*quantiles, 0.5]
    quantiles = sorted(quantiles)

    # Add cluster labels and sample weights to data
    data = (
        pd.Series(series)
        .to_frame(name="var")
        .pipe(lambda x: turn_cats_to_continuous(x, ["var"]) if x["var"].dtype == "category" else x)
        .assign(**{"cluster": pd.Series(labels).astype("category")})
        .pipe(lambda x: x if sample_weight is None else x.assign(**{"weight": sample_weight}))
    )

    # Get statistics of the variable for each cluster and overall
    agg_spec = {
        "mean": ("var", partial_fn_with_info(np.average, ignore_nan=True, weights=getattr(data, "weight", None))),
        "std": ("var", partial_fn_with_info(std_weighted, ignore_nan=True, weights=getattr(data, "weight", None))),
        **{
            f"q{q:g}": (
                "var",
                partial_fn_with_info(
                    lambda x, q=q, **kws: np.float64(np.quantile(x, q, method="inverted_cdf", **kws)),
                    ignore_nan=True,
                    weights=getattr(data, "weight", None),
                ),
            )
            for q in quantiles
        },
        "n": ("var", "count") if sample_weight is None else ("weight", "sum"),
    }
    statistics = agg_with_overall(data, "cluster", **agg_spec, overall_label="all", observed=False).assign(
        **{
            "n_str": lambda x: (
                x["n"]
                .round(cluster_size_decimals)
                .pipe(lambda y: y.astype("int") if sample_weight is None or cluster_size_decimals <= 0 else y)
                .astype("str")
            )
        }
    )

    # Get data for plotting quantile bands
    data_ggplot = (
        statistics.reset_index()
        .drop(columns=["mean", "std", "n", "n_str"])
        .pipe(lambda x: pd.wide_to_long(x, "q", i="cluster", j="quantile", sep="", suffix=r"[\d.]+"))
        .rename(columns={"q": "value"})
        .reset_index()
        .assign(
            **{
                "quantile_next": lambda x: x.groupby("cluster", observed=False)["quantile"].shift(-1),
                "value_next": lambda x: x.groupby("cluster", observed=False)["value"].shift(-1),
            }
        )
        .dropna(subset=["quantile_next", "value_next"])
    )

    # X axis ticks ordered
    x_axis_ordering = statistics.pipe(
        lambda x: x.sort_values(["mean", "std"], ascending=(ascending, not ascending)) if sort else x
    ).index.to_list()

    # Make legend labels for quantile bands
    legend_labels = {x: f"q{x}–q{quantiles[i + 1]}" for i, x in enumerate(quantiles[:-1])}

    # Construct plot
    return (
        p9.ggplot(data_ggplot)
        + p9.aes(x="cluster", xend="cluster", y="value", yend="value_next", color="factor(quantile, ordered=True)")
        + p9.geom_segment(size=column_size)
        + p9.geom_point(
            p9.aes(x="cluster", y="q0.5"), statistics.reset_index(), shape="_", inherit_aes=False, size=column_size - 1
        )
        + p9.geom_text(
            p9.aes(x="cluster", y=f"q{np.max(quantiles):g}", label="n_str"),
            statistics.reset_index(),
            inherit_aes=False,
            va="bottom",
            nudge_y=0.03 * (data["var"].max() - data["var"].min()),
            size=8,
        )
        + p9.labs(y=str(series.name))
        + p9.scale_x_discrete(limits=x_axis_ordering)
        + p9.scale_color_cmap_d(cmap_name="viridis", name="Quantile band", labels=legend_labels)  # type: ignore
        + theme_ggplot
    )


def make_plot_histogram_by_cluster(
    series: ArrayLike,
    labels: ArrayLike,
    sample_weight: ArrayLike | None = None,
    bins: int | ArrayLike = 15,
    cluster_size_decimals: int = 0,
    clusters: Iterable[Any] | None = None,
) -> p9.ggplot:
    """Plot within-cluster histogram proportions, with optional survey weights."""
    series = pd.Series(series)
    data = (
        series.to_frame("var")
        .pipe(lambda x: turn_cats_to_continuous(x, ["var"]) if x["var"].dtype == "category" else x)
        .assign(
            var=lambda x: pd.to_numeric(x["var"]),
            cluster=pd.Series(labels).astype("category"),
            weight=1 if sample_weight is None else np.asarray(sample_weight),
        )
        .dropna(subset=["var", "cluster", "weight"])
    )
    data = data.loc[np.isfinite(data["var"])]
    if data.empty:
        return (
            p9.ggplot()
            + p9.annotate("text", x=0, y=0, label="No observed values")
            + p9.labs(x=str(series.name))
            + theme_ggplot
            + p9.theme(axis_text=p9.element_blank(), axis_ticks=p9.element_blank(), panel_grid=p9.element_blank())
        )

    bin_edges = np.histogram_bin_edges(data["var"].to_numpy(dtype=float), bins=bins)
    statistics = agg_with_overall(data, "cluster", n=("weight", "sum"), overall_label="all", observed=True)
    decimals = 0 if sample_weight is None else cluster_size_decimals
    cluster_names = {cluster: f"{cluster}  ({n:.{decimals}f})" for cluster, n in statistics["n"].items()}
    cluster_order = statistics.index.to_list()
    if clusters is not None:
        cluster_order = [cluster for cluster in clusters if cluster in statistics.index] + ["all"]
    data_ggplot = (
        pd.concat([data, data.assign(cluster="all")], ignore_index=True)
        .loc[lambda x: x["cluster"].isin(cluster_order)]
        .assign(
            cluster=lambda x: pd.Categorical(x["cluster"], categories=cluster_order).rename_categories(cluster_names)
        )
    )

    return (
        p9.ggplot(data_ggplot)
        + p9.aes(x="var", weight="weight", y=p9.after_stat("density * width"))
        + p9.geom_histogram(breaks=bin_edges, fill="#595959", color="white", size=0.1)
        + p9.facet_wrap("cluster")
        + p9.labs(x=str(series.name), y="Proportion within cluster")
        + p9.scale_y_continuous(labels=lambda values: [f"{value:.0%}" for value in values])
        + theme_ggplot
    )


def make_plot_categorical_var_dist_by_cluster(
    series: ArrayLike,
    labels: ArrayLike,
    sample_weight: ArrayLike | None = None,
    cluster_size_decimals: int = 0,
    value_labels: Mapping[Any, Any] | None = None,
    clusters: Iterable[Any] | None = None,
) -> p9.ggplot:
    """Creates the plot of the distribution of a categorical variable in each cluster."""
    series = pd.Series(series)

    # Add cluster labels and sample weights to data
    data = (
        series.to_frame("var")
        .assign(**{"cluster": pd.Series(labels).astype("category")})
        .pipe(lambda x: x if sample_weight is None else x.assign(**{"weight": sample_weight}))
    )

    # Get statistics of the variable for each cluster and overall
    statistics = agg_with_overall(
        data,
        "cluster",
        **{"n": ("var", "count") if sample_weight is None else ("weight", "sum")},
        overall_label="all",
        observed=False,
    ).assign(
        **{
            "n_str": lambda x: (
                x["n"]
                .round(cluster_size_decimals)
                .pipe(lambda y: y.astype("int") if sample_weight is None or cluster_size_decimals <= 0 else y)
                .astype("str")
            )
        }
    )

    # Get data for plotting value counts
    data_ggplot = (
        agg_with_overall(
            data,
            "cluster",
            **{"n": ("var", "count") if sample_weight is None else ("weight", "sum")},
            by_vars=["cluster", "var"],
            overall_label="all",
            observed=False,
        )
        .reset_index()
        .assign(
            **{
                "proportion": lambda x: x["n"].div(statistics.loc[x["cluster"], "n"].values),
            }
        )
    )
    if clusters is not None:
        cluster_order = [cluster for cluster in clusters if cluster in statistics.index] + ["all"]
        data_ggplot = data_ggplot.loc[lambda x: x["cluster"].isin(cluster_order)].assign(
            cluster=lambda x: x["cluster"].cat.set_categories(cluster_order)
        )
    data_ggplot["cluster"] = data_ggplot["cluster"].cat.rename_categories(
        lambda cluster: f"{cluster}  ({statistics.loc[cluster, 'n_str']})"
    )

    # Construct plot
    return (
        p9.ggplot(data_ggplot)
        + p9.aes(x="var", y="proportion")
        + p9.geom_bar(stat=p9.stat_identity())
        + p9.coord_flip()
        + p9.facet_wrap("cluster")
        + p9.labs(x=str(series.name))
        + p9.scale_x_discrete(
            limits=lambda x: list(reversed(x)), labels=True if value_labels is None else dict(value_labels), drop=False
        )
        + theme_ggplot
        + p9.theme(axis_text_x=p9.element_text(rotation=270))
    )
