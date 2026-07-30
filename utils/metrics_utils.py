"""Module with utilities for clustering metrics."""

from __future__ import annotations

import functools
import math

import numpy as np
import pandas as pd
import validclust
from numpy.typing import ArrayLike
from scipy.sparse import issparse
from scipy.stats import chi2_contingency, entropy
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.metrics.cluster._unsupervised import check_number_of_labels
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_chunked
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import _safe_indexing, check_X_y
from sklearn.utils._array_api import _atol_for_type
from sklearn.utils.validation import _check_sample_weight

from utils.misc_utils import contingency_matrix


# Copied from sklearn and modified to enable sample weights
def pair_confusion_matrix(labels_true, labels_pred, sample_weight: ArrayLike | None = None):
    """Pair confusion matrix arising from two clusterings [1]_."""
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, labels_true, dtype=np.float64, only_non_negative=True)
    n_samples = np.float64(labels_true.shape[0]) if sample_weight is None else np.sum(sample_weight)
    sum_squares_weight = n_samples if sample_weight is None else np.dot(sample_weight, sample_weight)

    contingency = contingency_matrix(
        labels_true, labels_pred, sample_weight=sample_weight, sparse=True, dtype=np.float64
    )
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency.data**2).sum()
    C = np.empty((2, 2), dtype=np.float64)
    C[1, 1] = sum_squares - sum_squares_weight
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples**2 - C[0, 1] - C[1, 0] - sum_squares
    return C


# Copied from sklearn and modified to enable sample weights
def adjusted_rand_score(labels_true, labels_pred, sample_weight: ArrayLike | None = None):
    """Rand index adjusted for chance."""
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred, sample_weight=sample_weight)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    return 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


# Copied from sklearn and modified to enable sample weights
def silhouette_score(X, labels, *, sample_weight: ArrayLike | None = None, metric="euclidean", **kwds):
    """Compute the mean Silhouette Coefficient of all samples."""
    return np.average(
        silhouette_samples(X, labels, sample_weight=sample_weight, metric=metric, **kwds), weights=sample_weight
    )


# Copied from sklearn and modified to enable sample weights
def _silhouette_reduce(D_chunk, start, labels, label_freqs, sample_weight):
    """Accumulate silhouette statistics for vertical chunk of X."""
    n_chunk_samples = D_chunk.shape[0]
    # accumulate distances from each sample to each cluster
    cluster_distances = np.zeros((n_chunk_samples, len(label_freqs)), dtype=D_chunk.dtype)

    if issparse(D_chunk):
        if D_chunk.format != "csr":
            raise TypeError("Expected CSR matrix. Please pass sparse matrix in CSR format.")
        for i in range(n_chunk_samples):
            indptr = D_chunk.indptr
            indices = D_chunk.indices[indptr[i] : indptr[i + 1]]
            distances = D_chunk.data[indptr[i] : indptr[i + 1]]
            sample_labels = np.take(labels, indices)
            weights = distances if sample_weight is None else distances * sample_weight[indices]
            cluster_distances[i] += np.bincount(sample_labels, weights=weights, minlength=len(label_freqs))
    else:
        for i in range(n_chunk_samples):
            distances = D_chunk[i]
            sample_labels = labels
            weights = distances if sample_weight is None else distances * sample_weight
            cluster_distances[i] += np.bincount(sample_labels, weights=weights, minlength=len(label_freqs))

    # intra_index selects intra-cluster distances within cluster_distances
    end = start + n_chunk_samples
    intra_index = (np.arange(n_chunk_samples), labels[start:end])
    # intra_cluster_distances are averaged over cluster size outside this function
    intra_cluster_distances = cluster_distances[intra_index]
    # of the remaining distances we normalise and extract the minimum
    cluster_distances[intra_index] = np.inf
    cluster_distances /= label_freqs
    inter_cluster_distances = cluster_distances.min(axis=1)
    return intra_cluster_distances, inter_cluster_distances


# Copied from sklearn and modified to enable sample weights
# ruff: disable[B905]
def silhouette_samples(X, labels, *, sample_weight: ArrayLike | None = None, metric="euclidean", **kwds):
    """Compute the Silhouette Coefficient for each sample."""
    X, labels = check_X_y(X, labels, accept_sparse=["csr"])
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64, only_non_negative=True)

    # Check for non-zero diagonal entries in precomputed distance matrix
    if metric == "precomputed":
        error_msg = ValueError(
            "The precomputed distance matrix contains non-zero elements on the diagonal. Use np.fill_diagonal(X, 0)."
        )
        if X.dtype.kind == "f":
            atol = _atol_for_type(X.dtype)

            if np.any(np.abs(X.diagonal()) > atol):
                raise error_msg
        elif np.any(X.diagonal() != 0):  # integral dtype
            raise error_msg

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels, weights=sample_weight)
    check_number_of_labels(len(le.classes_), n_samples)

    kwds["metric"] = metric
    reduce_func = functools.partial(
        _silhouette_reduce, labels=labels, label_freqs=label_freqs, sample_weight=sample_weight
    )
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func, **kwds))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)

    denom = (
        label_freqs.take(labels, mode="clip") - 1
        if sample_weight is None
        else label_freqs.take(labels, mode="clip") - sample_weight
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom

    sil_samples = inter_clust_dists - intra_clust_dists
    with np.errstate(divide="ignore", invalid="ignore"):
        sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples)


# ruff: enable[B905]


# Copied from sklearn and modified to enable sample weights
def calinski_harabasz_score(X, labels, sample_weight: ArrayLike | None = None):
    """Compute the Calinski and Harabasz score."""
    X, labels = check_X_y(X, labels)
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64, only_non_negative=True)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples = len(X) if sample_weight is None else np.sum(sample_weight)
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, len(X))

    extra_disp, intra_disp = 0.0, 0.0
    mean = np.average(X, weights=sample_weight, axis=0)
    for k in range(n_labels):
        mask_k = labels == k
        cluster_k = X[mask_k]
        weights_k = None if sample_weight is None else sample_weight[mask_k]
        n_k = len(cluster_k) if weights_k is None else np.sum(weights_k)
        mean_k = np.average(cluster_k, weights=weights_k, axis=0)
        extra_disp += n_k * np.sum((mean_k - mean) ** 2)
        intra_disp += (
            np.sum((cluster_k - mean_k) ** 2)
            if weights_k is None
            else np.sum(weights_k[:, np.newaxis] * (cluster_k - mean_k) ** 2)
        )
    return 1.0 if intra_disp == 0.0 else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))


# Copied from sklearn and modified to enable sample weights
def davies_bouldin_score(X, labels, sample_weight: ArrayLike | None = None):
    """Compute the Davies-Bouldin score."""
    X, labels = check_X_y(X, labels)
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64, only_non_negative=True)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, len(X))

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=float)
    for k in range(n_labels):
        mask_k = labels == k
        cluster_k = _safe_indexing(X, mask_k)
        weights_k = None if sample_weight is None else sample_weight[mask_k]
        centroid = np.average(cluster_k, weights=weights_k, axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid]), weights=weights_k, axis=0).item()

    centroid_distances = pairwise_distances(centroids)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)


# Copied from sklearn and modified to enable sample weights
def normalized_mutual_info_score(
    labels_true, labels_pred, *, sample_weight: ArrayLike | None = None, average_method="arithmetic"
):
    """Normalized Mutual Information between two clusterings."""
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, labels_true, dtype=np.float64, only_non_negative=True)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Special limit cases: no clustering since the data is not split.
    # It corresponds to both labellings having zero entropy.
    # This is a perfect match hence return 1.0.
    if classes.shape[0] == clusters.shape[0] == 1 or classes.shape[0] == clusters.shape[0] == 0:
        return 1.0

    contingency = contingency_matrix(
        labels_true, labels_pred, sample_weight=sample_weight, sparse=True, dtype=np.float64
    )
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)

    # At this point mi = 0 can't be a perfect match (the special case of a single
    # cluster has been dealt with before). Hence, if mi = 0, the nmi must be 0 whatever
    # the normalization.
    if mi == 0:
        return 0.0

    # Calculate entropy for each labeling
    pi = np.asarray(contingency.sum(axis=1)).ravel()
    pj = np.asarray(contingency.sum(axis=0)).ravel()
    h_true, h_pred = entropy(pi), entropy(pj)

    normalizer = _generalized_average(h_true, h_pred, average_method)
    return mi / normalizer


def variation_of_information(
    labels_true: ArrayLike, labels_pred: ArrayLike, sample_weight: ArrayLike | None = None
) -> float:
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, labels_true, dtype=np.float64, only_non_negative=True)

    contingency = contingency_matrix(
        labels_true, labels_pred, sample_weight=sample_weight, sparse=True, dtype=np.float64
    )

    pi = np.asarray(contingency.sum(axis=1)).ravel()
    pj = np.asarray(contingency.sum(axis=0)).ravel()
    h_true, h_pred = entropy(pi), entropy(pj)

    voi = h_true + h_pred - 2 * mutual_info_score(labels_true, labels_pred, contingency=contingency)
    return voi.item()


# Copied from scipy and modified to enable sample weights
def association(observed, method="cramer", correction=False, lambda_=None):
    """Calculates degree of association between two nominal variables."""
    arr = np.asarray(observed)

    if len(arr.shape) != 2:
        raise ValueError("method only accepts 2d arrays")

    chi2_stat = chi2_contingency(arr, correction=correction, lambda_=lambda_)

    phi2 = chi2_stat.statistic / arr.sum()
    n_rows, n_cols = arr.shape
    if method == "cramer":
        value = phi2 / min(n_cols - 1, n_rows - 1)
    elif method == "tschuprow":
        value = phi2 / math.sqrt((n_rows - 1) * (n_cols - 1))
    elif method == "pearson":
        value = phi2 / (1 + phi2)
    else:
        raise ValueError("Invalid argument value: 'method' argument must be 'cramer', 'tschuprow', or 'pearson'")

    return math.sqrt(value)


def dunn_index(data: pd.DataFrame, labels: pd.Series[pd.CategoricalDtype]) -> np.float64:
    """Calculate the Dunn CVI."""
    dist = pairwise_distances(data)
    return validclust.dunn(dist, labels)
