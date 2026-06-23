"""Module with utilities for bootstrapping."""

import itertools
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_array, check_consistent_length, resample
from sklearn.utils.validation import _check_sample_weight

from utils.metrics_utils import adjusted_rand_score


def _compute_within_metrics(
    estimator: BaseEstimator,
    X: np.typing.ArrayLike,
    within_metrics: Sequence[Callable[..., Any]],
    sample_weight: np.typing.ArrayLike | None = None,
) -> list[Any]:
    """Compute metrics for one estimator."""
    X = check_array(X, force_all_finite=False)
    metric_kwargs = {}
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)
        metric_kwargs["sample_weight"] = sample_weight

    scores = []
    for metric in within_metrics:
        res = metric(estimator, X, **metric_kwargs)
        scores.append(res)
    return scores


def _compute_between_metrics(
    cluster_preds: list[np.typing.ArrayLike],
    between_metrics: Sequence[Callable[..., Any]],
    sample_weight: np.typing.ArrayLike | None = None,
) -> np.typing.NDArray:
    """Compute mean and standard deviation of metrics comparing all pairs of clusterings."""
    for i, x in enumerate(cluster_preds):
        cluster_preds[i] = check_array(x, ensure_2d=False)
    metric_kwargs = {}
    if sample_weight is not None:
        sample_weight = check_array(sample_weight, ensure_2d=False)
        check_consistent_length(cluster_preds[0], sample_weight)
        metric_kwargs["sample_weight"] = sample_weight

    scores = []
    for metric in between_metrics:
        scores_single = []
        for x, y in itertools.combinations(cluster_preds, r=2):
            res = metric(x, y, **metric_kwargs)
            scores_single.append(res)
        scores.append([np.mean(scores_single), np.std(scores_single)])
    return np.array(scores)


def _make_summary_pandas(scores: np.typing.ArrayLike) -> pd.DataFrame:
    return (
        pd.DataFrame(scores, columns=pd.Index(["mean", "std"]))
        .reset_index(names="metric")
        .assign(**{"metric": lambda x: x["metric"].astype("category")})
    )


def bootstrap(
    estimator: BaseEstimator,
    X: np.typing.ArrayLike,
    n_boot: int = 10,
    within_metrics: Sequence[Callable[..., Any]] | None = None,
    between_metrics: Sequence[Callable[..., Any]] | None = None,
    sample_weight: np.typing.ArrayLike | None = None,
    rng: np.random.Generator | np.random.BitGenerator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bootstrap an iterator and calculate metrics.

    `within_metrics` are calculated for each iteration with signature `(estimator, X) -> float` if `sample_weight` is
    `None` or `(estimator, X, sample_weight) -> float` if `sample_weight` is not `None`. `between_metrics` are
    calculated for any pair of prediction label vectors with signature `(labels_pred0, labels_pred1) -> float` if
    `sample_weight` is `None` or `(labels_pred0, labels_pred1, sample_weight) -> float` if `sample_weight` is not
    `None`. For both `within_metrics` and `between_metrics` the (weighted) mean and (weighted) standard deviation are
    returned.
    """
    X = check_array(X, force_all_finite=False)
    metric_kwargs = {}
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)
        metric_kwargs["sample_weight"] = sample_weight

    if within_metrics is None:
        within_metrics = (lambda estimator, X, **kwargs: estimator.score(X, **kwargs),)
    if between_metrics is None:
        between_metrics = (adjusted_rand_score,)

    rng = np.random.default_rng(rng)

    # Do bootstrapping and compute within metrics
    within_scores = []
    cluster_preds = []
    for _ in range(n_boot):
        estimator_boot = clone(estimator)
        idx_boot = resample(
            np.arange(len(X)), replace=True, n_samples=len(X), random_state=np.random.RandomState(rng.bit_generator)
        )
        X_boot = X[idx_boot]

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight[idx_boot]
        estimator_boot.fit(X_boot, **fit_kwargs)
        within_single = _compute_within_metrics(estimator_boot, X, within_metrics, **metric_kwargs)
        within_scores.append(within_single)

        cluster_pred = estimator_boot.predict(X)
        cluster_preds.append(cluster_pred)
    within_scores = np.stack((np.mean(within_scores, axis=0), np.std(within_scores, axis=0)), axis=1)
    within_scores = _make_summary_pandas(within_scores)

    # Compare clusterings using between metrics
    between_scores = _compute_between_metrics(cluster_preds, between_metrics, **metric_kwargs)
    between_scores = _make_summary_pandas(between_scores)

    return within_scores, between_scores


def bootstrap_grid_search(
    estimator: BaseEstimator,
    X: np.typing.ArrayLike,
    param_grid: dict | Iterable[dict],
    n_boot: int = 10,
    within_metrics: Sequence[Callable[..., Any]] | None = None,
    between_metrics: Sequence[Callable[..., Any]] | None = None,
    sample_weight: np.typing.ArrayLike | None = None,
    rng: np.random.Generator | np.random.BitGenerator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iteratively bootstrap an iterator and calculate metrics for a parameter grid.

    `within_metrics` are calculated for each iteration with signature `(estimator, X) -> float` if `sample_weight` is
    `None` or `(estimator, X, sample_weight) -> float` if `sample_weight` is not `None`. `between_metrics` are
    calculated for any pair of prediction label vectors with signature `(labels_pred0, labels_pred1) -> float` if
    `sample_weight` is `None` or `(labels_pred0, labels_pred1, sample_weight) -> float` if `sample_weight` is not
    `None`. For both `within_metrics` and `between_metrics` the (weighted) mean and (weighted) standard deviation are
    returned.
    """
    rng = np.random.default_rng(rng)

    results = []
    param_grid = ParameterGrid(param_grid)
    n_comb = len(param_grid)
    for i, params in enumerate(param_grid):
        print(f"{i}/{n_comb}", end="\r")
        estimator_inner = clone(estimator).set_params(**params)
        within_scores, between_scores = bootstrap(
            estimator_inner,
            X,
            n_boot=n_boot,
            within_metrics=within_metrics,
            between_metrics=between_metrics,
            sample_weight=sample_weight,
            rng=rng,
        )

        within_scores = within_scores.assign(**params)
        between_scores = between_scores.assign(**params)
        results.append((within_scores, between_scores))
    print(f"{n_comb}/{n_comb}")

    return tuple(pd.concat(x, ignore_index=True) for x in zip(*results))
