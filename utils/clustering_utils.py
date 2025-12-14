"""Module with functions for clustering.ipynb"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import plotnine as p9
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.metrics import silhouette_samples
from umap import UMAP

from utils.utils import theme_ggplot

pd.options.mode.copy_on_write = True

# Create projection method types, lists, and
DeterministicProjection = Isomap
deterministic_projection_list = [Isomap]
str_to_deterministic_projection: dict[str, type[DeterministicProjection]] = {
    x.__name__.lower(): x for x in deterministic_projection_list
}

RandomProjection = LocallyLinearEmbedding | MDS | SpectralEmbedding | TSNE | UMAP
random_projection_list = [LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE, UMAP]
str_to_random_projection: dict[str, type[RandomProjection]] = {x.__name__.lower(): x for x in random_projection_list}

Projection = DeterministicProjection | RandomProjection
projection_list = [*deterministic_projection_list, *random_projection_list]
str_to_projection: dict[str, type[Projection]] = {x.__name__.lower(): x for x in projection_list}


def reduce_dimen_2d(data: pd.DataFrame, projection_method: str | type[Projection] | None = None, **kwargs):
    """Projects data to 2d.

    The data is projected with the method specified as projection_method, if left unspecified the first two coordinates
    of the data are taken as the 2d representation. Valid projection methods are Isomap, LocallyLinearEmbedding, MDS,
    SpectralEmbedding, and TSNE. Certain projection methods require random numbers, otherwise a given RNG is ignored.
    """
    if projection_method is None:
        model_type = None
    elif isinstance(projection_method, str):
        # Get model from string
        model_type = str_to_projection[projection_method.lower()]
    elif issubclass(projection_method, Projection):
        model_type = projection_method

    if model_type is None:
        # Reduce to first two variables
        data_2d = data[:, :2].to_numpy()
    else:
        rng = np.random.default_rng(kwargs.pop("rng", None))
        if issubclass(model_type, RandomProjection):
            # Set random state in keyword argument dictionary
            kwargs["random_state"] = np.random.RandomState(rng.bit_generator)
        # Reduce dimensionality
        transform_2d = model_type(n_components=2, n_jobs=-1, **kwargs)
        data_2d = transform_2d.fit_transform(data)
        data_2d = typing.cast(np.typing.NDArray, data_2d)

    return pd.DataFrame({"x": data_2d[:, 0], "y": data_2d[:, 1]})


def plot_2d_data(data: pd.DataFrame, cluster_labels: pd.Series[pd.CategoricalDtype] | None = None):
    """
    Plots 2d dataset.

    The points are colored based on cluster_labels, if provided.
    """
    # Copy data so the original object is not modified
    data = data.copy()

    # Color based on clustering if provided
    aes_kwargs = {}
    if cluster_labels is not None:
        data["cluster"] = cluster_labels
        aes_kwargs |= dict(fill="cluster", color="cluster")

    # Create plot
    plot = (
        p9.ggplot(data)
        + p9.aes(x=data.columns[0], y=data.columns[1], **aes_kwargs)
        + p9.geom_point(alpha=0.8)
        + theme_ggplot
    )
    return plot


def print_clustering_info(cluster_labels: pd.Series[pd.CategoricalDtype]) -> None:
    """Prints some information about a clustering."""
    print(f"{len(cluster_labels.cat.categories.values)} clusters, {100 * cluster_labels.isna().mean():0.2f}% outliers")
    print(
        f"Percentage in clusters: {
            (cluster_labels.value_counts(normalize=True, dropna=False).apply(lambda x: 100 * x).sort_index().to_dict())
        }"
    )


def plot_silhouette(data: pd.DataFrame, cluster_labels: pd.Series[pd.CategoricalDtype]) -> p9.ggplot:
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
