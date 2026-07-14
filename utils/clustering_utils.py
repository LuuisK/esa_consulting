"""Module with utilities for clustering."""

from __future__ import annotations

from collections.abc import Sequence

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
from utils.misc_utils import (
    agg_with_overall,
    contingency_matrix,
    partial_fn_with_info,
    std_weighted,
    turn_cats_to_continuous,
    value_counts,
)
from utils.saving_utils import DatasetBundle

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


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
    data: DatasetBundle, labels: pd.Series[pd.CategoricalDtype], sample_weight: np.typing.ArrayLike | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster summaries.

    Returns a tuple containing means and standard deviations for continuous, binary, and ordinal variables, and counts
    for nominal variables. The indices of data and labels have to line up.
    """
    # Don't modify the input
    data = data.copy()

    if data.meta is None:
        raise ValueError("'data.meta' may not be None.")

    # Convert and split data types
    treat_cat_as_continuous = data.meta.loc[lambda x: x["type"].isin(["binary", "ordinal"]), "name"]
    data_with_converted_types = turn_cats_to_continuous(data.data, treat_cat_as_continuous)
    data_not_nominal = data_with_converted_types.select_dtypes(exclude="category")
    data_nominal = data_with_converted_types.select_dtypes(include="category")

    # Make summaries for continuous, binary, and ordinal variables
    agg_weights = None if sample_weight is None else pd.Series(sample_weight, index=data_with_converted_types.index)
    cluster_summaries_not_nominal = agg_with_overall(
        data_not_nominal.assign(**{"cluster": labels}),
        "cluster",
        [
            ("mean", partial_fn_with_info(np.average, ignore_nan=True, weights=agg_weights)),
            ("std", partial_fn_with_info(std_weighted, ignore_nan=True, weights=agg_weights)),
        ],
        overall_label="all",
    )

    # Make summaries for nominal variables
    cluster_dtype = labels.cat.add_categories("all").dtype
    index_summaries_nominal = (
        pd.DataFrame(
            [
                (cluster, col, value)
                for cluster in cluster_dtype.categories
                for col in data_nominal.columns
                for value in data_nominal[col].cat.categories
            ],
            columns=pd.Index(["cluster", "variable", "value"]),
        )
        .assign(**{"cluster": lambda x: x["cluster"].astype(cluster_dtype)})
        .pipe(pd.MultiIndex.from_frame)
    )
    cluster_summaries_nominal = (
        agg_with_overall(
            (
                data_nominal.assign(**{"cluster": labels})
                .pipe(lambda x: x if sample_weight is None else x.assign(**{"weight": sample_weight}))
                .melt(
                    id_vars="cluster" if sample_weight is None else ["cluster", "weight"],
                    value_vars=data_nominal.columns,
                    var_name="variable",
                    value_name="value",
                )
            ),
            "cluster",
            **{"n": ("cluster", "count") if sample_weight is None else ("weight", "sum")},
            by_vars=["cluster", "variable", "value"],
            overall_label="all",
            observed=True,
        )
        .reindex(index_summaries_nominal, fill_value=0)
        .reset_index()
    )

    return cluster_summaries_not_nominal, cluster_summaries_nominal


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
        print(
            value_counts(cluster_labels, sort=True, normalize=True, dropna=False, sample_weight=sample_weight)
            .apply(lambda x: f"{100 * x:.2f}%")
            .to_frame()
            .reset_index(names="cluster")
            .to_dict()
        )
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
