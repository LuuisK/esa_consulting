"""Module with utilities for clustering."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pandas._typing as pdt
import validclust
from gapstat import gapstat_score
from scipy.stats import entropy
from scipy.stats.contingency import association, chi2_contingency
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    mutual_info_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from utils.dimensionality_reduction_utils import BlockPCA

pd.options.mode.copy_on_write = True


def print_clustering_info(cluster_labels: pd.Series[pd.CategoricalDtype], sort_by_size: bool = False) -> None:
    """Prints some information about a clustering."""
    print(f"{len(cluster_labels.cat.categories.values)} clusters, {100 * cluster_labels.isna().mean():0.2f}% outliers")
    cluster_frequencies = cluster_labels.value_counts(normalize=True, ascending=False, dropna=False)
    cluster_frequencies = cluster_frequencies if sort_by_size else cluster_frequencies.sort_index()
    print(f"Percentage in clusters: {cluster_frequencies.apply(lambda x: 100 * x).to_dict()}")


def predict_clustering_labels(model: BaseEstimator, data: pd.DataFrame) -> pd.Series[pd.CategoricalDtype]:
    """Predicts labels for a dataset using a clustering model."""
    return (
        pd.Series(data=model.predict(data), index=data.index.copy())  # type: ignore
        .replace(-1, pd.NA)
        .astype("category")
    )


def predict_clustering_proba(model: BaseEstimator, data: pd.DataFrame) -> pd.DataFrame:
    """Predicts cluster probabilities for a dataset using a clustering model."""
    return pd.DataFrame(data=model.predict_proba(data), index=data.index.copy()).rename(  # type: ignore
        columns=lambda x: f"cluster_{x}"
    )


def get_cluster_means(model: GaussianMixture, pca: BlockPCA) -> pd.DataFrame:
    """Backtransform cluster means from PCA space into variable space."""
    columns = (f"{group}_comp_{i}" for group in pca.var_groups for i in range(pca.n_components_per_group[group]))
    cluster_means = pd.DataFrame(model.means_).rename(columns=lambda x: next(columns))  # type: ignore
    return pca.inverse_transform(cluster_means).rename_axis("cluster", axis=0)


def get_cluster_vars(model: GaussianMixture, pca: BlockPCA) -> tuple[pd.DataFrame, ...]:
    """Backtransform cluster variances from PCA space into variable space."""
    model_gmm = model

    vars_in_order = list(itertools.chain.from_iterable(pca.var_groups.values()))
    return tuple(
        pd.DataFrame(var_cluster).rename(
            index=lambda x, iter=iter(vars_in_order): next(iter), columns=lambda x, iter=iter(vars_in_order): next(iter)
        )
        for var_cluster in pca.components_.T @ model_gmm.covariances_ @ pca.components_  # type: ignore
    )


def cluster_summaries(
    data: pd.DataFrame, labels: pd.Series[pd.CategoricalDtype], agg_func: pdt.AggFuncTypeFrame = ["mean", "std"]
):
    """
    Cluster summaries

    The data must not contain categorical columns. The indices of data and labels have to line up.
    """
    return data.assign(**{"cluster": labels}).groupby("cluster", observed=True).aggregate(agg_func)


def dunn_index(data: pd.DataFrame, labels: pd.Series[pd.CategoricalDtype]) -> np.float64:
    dist = pairwise_distances(data)
    return validclust.dunn(dist, labels)


def print_cluster_statistics(
    cluster_labels_list: Sequence[pd.Series[pd.CategoricalDtype]],
    datas: Sequence[pd.DataFrame],
    labels: Sequence[str] | None = None,
) -> None:
    """Prints some statistics of one or more clusterings."""
    labels = [f"clustering {i}" for i in range(len(datas))] if labels is None else labels

    len_label = max([len(label) for label in labels])
    labels_padded = [f"{label + ':':<{len_label + 3}}" for label in labels]

    print(f"Comparison between {labels}:")
    print("")

    for cluster_labels, data, label_padded in zip(cluster_labels_list, datas, labels_padded, strict=True):
        print(label_padded.capitalize(), end="")
        print_clustering_info(cluster_labels, sort_by_size=True)
    print("")

    for cluster_labels, data, label_padded in zip(cluster_labels_list, datas, labels_padded, strict=True):
        print(f"Davies‒Bouldin index {label_padded}{davies_bouldin_score(data, labels=cluster_labels)}")
    print("")

    for cluster_labels, data, label_padded in zip(cluster_labels_list, datas, labels_padded, strict=True):
        print(f"Caliński–Harabasz index {label_padded}{calinski_harabasz_score(data, labels=cluster_labels)}")
    print("")

    for cluster_labels, data, label_padded in zip(cluster_labels_list, datas, labels_padded, strict=True):
        print(f"Mean silhouette coefficient {label_padded}{silhouette_score(data, labels=cluster_labels)}")
    print("")

    for cluster_labels, data, label_padded in zip(cluster_labels_list, datas, labels_padded, strict=True):
        print(f"Dunn index {label_padded}{dunn_index(data, labels=cluster_labels)}")
    print("")

    for cluster_labels, data, label_padded in zip(cluster_labels_list, datas, labels_padded, strict=True):
        print(f"Gap statistic {label_padded}{gapstat_score(data, labels=cluster_labels)}")


def variation_of_information(x: pd.Series, y: pd.Series) -> float:
    voi = np.array(entropy(x)) + np.array(entropy(y)) - 2 * mutual_info_score(x, y)
    return voi.item()


def print_cluster_stability(
    cluster_labels1: pd.Series[pd.CategoricalDtype], cluster_labels2: pd.Series[pd.CategoricalDtype]
) -> None:
    """Prints a comparison of two clusterings of the same dataset."""
    print(f"Adjusted rand index:       {adjusted_rand_score(cluster_labels1, cluster_labels2)}")
    print(f"Mutual information:        {normalized_mutual_info_score(cluster_labels1, cluster_labels2)}")
    print(f"Variation of information:  {variation_of_information(cluster_labels1, cluster_labels2)}")
    cont_table = pd.crosstab(cluster_labels1, cluster_labels2)
    print(f"Cramér's V:                {association(observed=cont_table, method='cramer')}")
    print(f"Chi squared statistic:     {chi2_contingency(observed=cont_table)[0]}")
