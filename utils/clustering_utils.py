"""Module with utilities for clustering."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
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

pd.options.mode.copy_on_write = True


def print_clustering_info(cluster_labels: pd.Series[pd.CategoricalDtype], sort_by_size: bool = False) -> None:
    """
    Prints some information about a clustering.
    """
    print(f"{len(cluster_labels.cat.categories.values)} clusters, {100 * cluster_labels.isna().mean():0.2f}% outliers")
    cluster_probabilities = cluster_labels.value_counts(normalize=True, ascending=False, dropna=False)
    cluster_probabilities = cluster_probabilities if sort_by_size else cluster_probabilities.sort_index()
    print(f"Percentage in clusters: {cluster_probabilities.apply(lambda x: 100 * x).to_dict()}")


def dunn_index(data: pd.DataFrame, labels: pd.Series[pd.CategoricalDtype]) -> np.float64:
    dist = pairwise_distances(data)
    return validclust.dunn(dist, labels)


def predict_clustering_labels(model: BaseEstimator, data: pd.DataFrame) -> pd.Series[pd.CategoricalDtype]:
    """
    Predicts labels for a dataset with a clustering model.
    """
    return (
        pd.Series(data=model.predict(data), index=data.index.copy())  # type: ignore
        .replace(-1, pd.NA)
        .astype("category")
    )


def print_cluster_statistics(
    cluster_labels_list: Sequence[pd.Series[pd.CategoricalDtype]],
    datas: Sequence[pd.DataFrame],
    labels: Sequence[str] | None = None,
) -> None:
    """
    Prints some statistics of one or more clusterings.
    """
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


def variation_of_information(x, y):
    voi = entropy(x) + entropy(y) - 2 * mutual_info_score(x, y)
    return voi


def print_cluster_stability(
    cluster_labels1: pd.Series[pd.CategoricalDtype], cluster_labels2: pd.Series[pd.CategoricalDtype]
) -> None:
    """
    Prints a comparison of two clusterings of the same dataset.
    """
    print(f"Adjusted rand index:       {adjusted_rand_score(cluster_labels1, cluster_labels2)}")
    print(f"Mutual information:        {normalized_mutual_info_score(cluster_labels1, cluster_labels2)}")
    print(f"Variation of information:  {variation_of_information(cluster_labels1, cluster_labels2)}")
    cont_table = pd.crosstab(cluster_labels1, cluster_labels2)
    print(f"Cramér's V:                {association(observed=cont_table, method='cramer')}")
    print(f"Chi squared statistic:     {chi2_contingency(observed=cont_table)[0]}")


def cluster_summaries(
    data: pd.DataFrame, labels: pd.Series[pd.CategoricalDtype], agg_func: pd._typing.AggFuncTypeFrame = ["mean", "std"]
):
    """
    Cluster summaries

    The data must not contain categorical columns. The indices of data and labels have to line up.
    """
    cluster_summaries = data.assign(**{"cluster": labels}).groupby("cluster", observed=True).aggregate(agg_func)
    return cluster_summaries
