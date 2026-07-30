"""Module with utilities for bootstrapping."""

import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from pandas._typing import Axes
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_array
from sklearn.utils.validation import _check_sample_weight

from utils.metrics_utils import adjusted_rand_score


def _make_summary_pandas(scores: dict[str, Any], columns: Axes | None) -> pd.DataFrame:
    return (
        pd.DataFrame.from_dict(scores, orient="index", columns=columns)
        .reset_index(names="metric")
        .assign(**{"metric": lambda x: x["metric"].astype("category")})
    )


def _compute_within_metrics_values(
    estimator: BaseEstimator,
    X: NDArray,
    cluster_pred: NDArray,
    within_metrics: Mapping[str, Callable[..., Any]],
    sample_weight: NDArray | None,
) -> pd.DataFrame:
    """Compute metrics for one estimator."""
    scores = {}
    for name, metric in within_metrics.items():
        kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
        scores[name] = metric(estimator, X, cluster_pred, **kwargs)
    return _make_summary_pandas(scores, columns=pd.Index(["value"]))


def _compute_between_metrics_summaries(
    cluster_preds: list[NDArray], between_metrics: Mapping[str, Callable[..., Any]], sample_weight: NDArray | None
) -> pd.DataFrame:
    """Compute mean and standard deviation of metrics comparing all pairs of clusterings."""
    scores: dict[str, Any] = {}
    for name, metric in between_metrics.items():
        scores_single = []
        for x, y in itertools.combinations(cluster_preds, r=2):
            kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
            res = metric(x, y, **kwargs)
            scores_single.append(res)
        scores_series = pd.Series(scores_single, dtype="float64")
        scores[name] = (scores_series.mean(), scores_series.std(ddof=1))
    return _make_summary_pandas(scores, columns=pd.Index(["mean", "std"]))


def _generate_boot_indices(
    n_samples: int, n_boot: int, sample_weight: NDArray | None, rng: np.random.Generator
) -> list[np.ndarray]:
    """Draw bootstrap indices from the empirical or weighted empirical distribution."""
    probabilities = None if sample_weight is None else sample_weight / sample_weight.sum(dtype=np.float64)
    return [rng.choice(n_samples, size=n_samples, replace=True, p=probabilities) for _ in range(n_boot)]


def bootstrap(
    estimator: BaseEstimator,
    X: ArrayLike,
    n_boot: int = 10,
    within_metrics: Mapping[str, Callable[..., Any]] | None = None,
    between_metrics: Mapping[str, Callable[..., Any]] | None = None,
    sample_weight: ArrayLike | None = None,
    rng: np.random.Generator | np.random.BitGenerator | None = None,
    boot_indices: Sequence[ArrayLike] | None = None,
) -> pd.DataFrame:
    """
    Bootstrap an iterator and calculate metrics.

    `within_metrics` are calculated for each iteration with signature `(estimator, X, cluster_pred) -> float` if
    `sample_weight` is `None` or `(estimator, X, cluster_pred, sample_weight) -> float` if `sample_weight` is not
    `None`. `between_metrics` are calculated for any pair of prediction label vectors with signature
    `(labels_pred0, labels_pred1) -> float` if `sample_weight` is `None` or
    `(labels_pred0, labels_pred1, sample_weight) -> float` if `sample_weight` is not `None`. For both `within_metrics`
    and `between_metrics` the mean and standard deviation over bootstrap iterations are returned.

    When `sample_weight` is supplied, observations are sampled with probability proportional to their weight. The
    resampled estimator is then fitted without weights to avoid applying the weights twice. Metrics evaluated on the
    original sample still receive `sample_weight`. This is a weighted empirical bootstrap, not a complex-survey
    bootstrap based on strata, clusters, or replicate weights.
    """
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2.")

    X = check_array(X, force_all_finite=False)
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64, only_non_negative=True)

    if within_metrics is None:
        within_metrics = {"score": lambda estimator, X, cluster_pred, **kwargs: estimator.score(X, **kwargs)}
    if between_metrics is None:
        between_metrics = {"ari": adjusted_rand_score}

    rng = np.random.default_rng(rng)
    boot_indices = _generate_boot_indices(len(X), n_boot, sample_weight, rng) if boot_indices is None else boot_indices
    boot_indices = check_array(boot_indices, ensure_2d=True, dtype=np.intp)
    if boot_indices.shape != (n_boot, len(X)) or np.any((boot_indices < 0) | (boot_indices >= len(X))):
        raise ValueError("boot_indices has the wrong shape.")

    # Do bootstrapping and compute within metrics
    within_scores = []
    cluster_preds: list[NDArray] = []
    for idx_boot in boot_indices:
        estimator_boot = clone(estimator)
        X_boot = X[idx_boot]

        # kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}  # TODO
        # estimator_boot.fit(X_boot, **kwargs)
        estimator_boot.fit(X_boot)

        cluster_pred = estimator_boot.predict(X)
        cluster_preds.append(cluster_pred)

        within_scores_single = _compute_within_metrics_values(
            estimator_boot, X, cluster_pred, within_metrics, sample_weight
        )
        within_scores.append(within_scores_single)
    within_summaries = (
        pd.concat(within_scores, ignore_index=True)
        .groupby("metric", observed=False)["value"]
        .agg(mean="mean", std=lambda x: x.std(ddof=1))
        .reset_index()
    )

    # Compare clusterings using between metrics
    between_summaries = _compute_between_metrics_summaries(cluster_preds, between_metrics, sample_weight)

    within_summaries["type"] = "within"
    between_summaries["type"] = "between"
    return pd.concat([within_summaries, between_summaries], ignore_index=True).assign(
        **{"type": lambda x: x["type"].astype(pd.CategoricalDtype(["within", "between"]))}
    )


def bootstrap_grid_search(
    estimator: BaseEstimator,
    X: ArrayLike,
    param_grid: dict | Iterable[dict],
    n_boot: int = 10,
    within_metrics: Mapping[str, Callable[..., Any]] | None = None,
    between_metrics: Mapping[str, Callable[..., Any]] | None = None,
    sample_weight: ArrayLike | None = None,
    rng: np.random.Generator | np.random.BitGenerator | None = None,
) -> pd.DataFrame:
    """
    Iteratively bootstrap an iterator and calculate metrics for a parameter grid.

    `within_metrics` are calculated for each iteration with signature `(estimator, X, cluster_pred) -> float` if
    `sample_weight` is `None` or `(estimator, X, cluster_pred, sample_weight) -> float` if `sample_weight` is not
    `None`. `between_metrics` are calculated for any pair of prediction label vectors with signature
    `(labels_pred0, labels_pred1) -> float` if `sample_weight` is `None` or
    `(labels_pred0, labels_pred1, sample_weight) -> float` if `sample_weight` is not `None`. For both `within_metrics`
    and `between_metrics` the mean and standard deviation over bootstrap iterations are returned.
    """
    X = check_array(X, force_all_finite=False)
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64, only_non_negative=True)
    rng = np.random.default_rng(rng)
    boot_indices = _generate_boot_indices(len(X), n_boot, sample_weight, rng)

    results = []
    param_grid = ParameterGrid(param_grid)
    n_comb = len(param_grid)
    for i, params in enumerate(param_grid):
        print(f"{i}/{n_comb}", end="\r")
        estimator_inner = clone(estimator).set_params(**params)
        metrics_summaries = bootstrap(
            estimator_inner,
            X,
            n_boot=n_boot,
            within_metrics=within_metrics,
            between_metrics=between_metrics,
            sample_weight=sample_weight,
            rng=rng,
            boot_indices=boot_indices,
        )

        metrics_summaries = metrics_summaries.assign(**params)
        results.append(metrics_summaries)
    print(f"{n_comb}/{n_comb}")

    return pd.concat(results, ignore_index=True)
