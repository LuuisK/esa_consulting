"""Module with utilities for clustering."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from gapstat import gapstat_score
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator
from sklearn.metrics import adjusted_rand_score

from utils.metrics_utils import (
    association,
    calinski_harabasz_score,
    davies_bouldin_score,
    dunn_index,
    normalized_mutual_info_score,
    silhouette_score,
    variation_of_information,
)
from utils.misc_utils import contingency_matrix, std_weighted

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


def print_clustering_info(
    cluster_labels: pd.Series[pd.CategoricalDtype],
    sort_by_size: bool = False,
    sample_weight: np.typing.ArrayLike | None = None,
) -> None:
    """Prints some information about a clustering."""
    print(f"{len(cluster_labels.cat.categories.values)} clusters, {100 * cluster_labels.isna().mean():0.2f}% outliers")
    if sample_weight is None:
        cluster_frequencies = cluster_labels.value_counts(normalize=True, ascending=False, dropna=False)
    else:
        cluster_frequencies = pd.Series(sample_weight).groupby(cluster_labels, dropna=False).sum()
        cluster_frequencies = (cluster_frequencies / cluster_frequencies.sum()).sort_values(ascending=False)
    if not sort_by_size:
        cluster_frequencies = cluster_frequencies.sort_index()
    print(f"Percentage in clusters: {cluster_frequencies.apply(lambda x: 100 * x).to_dict()}")


def predict_clustering_labels(model: BaseEstimator, data: pd.DataFrame) -> pd.Series[pd.CategoricalDtype]:
    """Predicts labels for a dataset using a clustering model."""
    return (
        pd.Series(model.predict(data), dtype="str")  # type: ignore
        .replace("-1", pd.NA)
        .pipe(lambda x: x.astype(pd.CategoricalDtype(pd.Series(x.unique()).sort_values(key=lambda x: x.astype(int)))))
    )


def predict_clustering_proba(model: BaseEstimator, data: pd.DataFrame) -> pd.DataFrame:
    """Predicts cluster probabilities for a dataset using a clustering model."""
    return pd.DataFrame(model.predict_proba(data)).rename(  # type: ignore
        columns=lambda x: f"cluster_{x}"
    )


def cluster_summaries(
    data: pd.DataFrame, labels: pd.Series[pd.CategoricalDtype], sample_weight: np.typing.ArrayLike | None = None
) -> pd.DataFrame:
    """
    Cluster summaries (mean and std).

    The data must not contain categorical columns. The indices of data and labels have to line up.
    """
    sample_weight = pd.Series(sample_weight, index=data.index)

    def weighted_agg(val: pd.Series, fn: Callable) -> Any:
        mask = val.notna()
        x = val[mask]
        w = sample_weight.loc[x.index]
        try:
            return fn(x.to_numpy(), weights=w.to_numpy())
        except ZeroDivisionError:
            return pd.NA

    def _summarize_data(data: pd.DataFrame) -> pd.Series:
        out = {}
        for col in data.columns:
            out[f"{col}_mean"] = weighted_agg(data[col], np.average)
            out[f"{col}_std"] = weighted_agg(data[col], std_weighted)
        return pd.Series(out)

    return (
        data.assign(**{"cluster": labels})
        .groupby("cluster", observed=True)
        .apply(_summarize_data, include_groups=False)
    )


def print_cluster_statistics(
    cluster_labels_list: Sequence[pd.Series[pd.CategoricalDtype]],
    datas: Sequence[pd.DataFrame],
    sample_weights: Sequence[np.typing.ArrayLike | None] | None = None,
    labels: Sequence[str] | None = None,
) -> None:
    """Prints some statistics of one or more clusterings."""
    if sample_weights is None:
        sample_weights = [None for i in range(len(datas))]

    if labels is None:
        labels = [f"clustering {i}" for i in range(len(datas))]

    len_label = max([len(label) for label in labels])
    labels_padded = [f"{label + ':':<{len_label + 3}}" for label in labels]

    print(f"Comparison between {labels}:")
    print("")

    for cluster_labels, data, sample_weight, label_padded in zip(
        cluster_labels_list, datas, sample_weights, labels_padded, strict=True
    ):
        print(label_padded.capitalize(), end="")
        print_clustering_info(cluster_labels, sort_by_size=True, sample_weight=sample_weight)
    print("")

    for cluster_labels, data, sample_weight, label_padded in zip(
        cluster_labels_list, datas, sample_weights, labels_padded, strict=True
    ):
        dbi = davies_bouldin_score(data, labels=cluster_labels, sample_weight=sample_weight)
        print(f"Davies‒Bouldin index {label_padded}{dbi}")
    print("")

    for cluster_labels, data, sample_weight, label_padded in zip(
        cluster_labels_list, datas, sample_weights, labels_padded, strict=True
    ):
        cal = calinski_harabasz_score(data, labels=cluster_labels, sample_weight=sample_weight)
        print(f"Caliński–Harabasz index {label_padded}{cal}")
    print("")

    for cluster_labels, data, sample_weight, label_padded in zip(
        cluster_labels_list, datas, sample_weights, labels_padded, strict=True
    ):
        sil = silhouette_score(data, labels=cluster_labels, sample_weight=sample_weight)
        print(f"Mean silhouette coefficient {label_padded}{sil}")
    print("")

    for cluster_labels, data, label_padded in zip(cluster_labels_list, datas, labels_padded, strict=True):
        di = dunn_index(data, labels=cluster_labels)
        print(f"Dunn index {label_padded}{di}")
    print("")

    for cluster_labels, data, label_padded in zip(cluster_labels_list, datas, labels_padded, strict=True):
        gap = gapstat_score(data, labels=cluster_labels)
        print(f"Gap statistic {label_padded}{gap}")


def print_cluster_stability(
    cluster_labels1: pd.Series[pd.CategoricalDtype],
    cluster_labels2: pd.Series[pd.CategoricalDtype],
    sample_weight: np.typing.ArrayLike | None = None,
) -> None:
    """Prints a comparison of two clusterings of the same dataset."""
    ari = adjusted_rand_score(cluster_labels1, cluster_labels2, sample_weight=sample_weight)
    nmi = normalized_mutual_info_score(cluster_labels1, cluster_labels2, sample_weight=sample_weight)
    vi = variation_of_information(cluster_labels1, cluster_labels2, sample_weight=sample_weight)

    cont_table = contingency_matrix(cluster_labels1, cluster_labels2, sample_weight=sample_weight)
    assoc = association(observed=cont_table, method="cramer")
    chi2_stat, *_ = chi2_contingency(observed=cont_table)

    print(f"Adjusted rand index:       {ari}")
    print(f"Mutual information:        {nmi}")
    print(f"Variation of information:  {vi}")
    print(f"Cramér's V:                {assoc}")
    print(f"Chi squared statistic:     {chi2_stat}")
