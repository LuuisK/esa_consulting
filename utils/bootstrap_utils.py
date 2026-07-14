"""Module with utilities for bootstrapping."""

import itertools
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from pandas._typing import Axes
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_array, check_consistent_length, resample
from sklearn.utils.validation import _check_sample_weight

from utils.metrics_utils import adjusted_rand_score


def _make_summary_pandas(scores: dict[str, Any], columns: Axes | None = None) -> pd.DataFrame:
    return (
        pd.DataFrame.from_dict(scores, orient="index", columns=columns)
        .reset_index(names="metric")
        .assign(**{"metric": lambda x: x["metric"].astype("category")})
    )


def _compute_within_metrics_values(
    estimator: BaseEstimator,
    X: np.typing.ArrayLike,
    within_metrics: Mapping[str, Callable[..., Any]],
    sample_weight: np.typing.ArrayLike | None = None,
) -> pd.DataFrame:
    """Compute metrics for one estimator."""
    X = check_array(X, force_all_finite=False)
    metric_kwargs = {}
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)
        metric_kwargs["sample_weight"] = sample_weight

    scores = {}
    for name, metric in within_metrics.items():
        scores[name] = metric(estimator, X, **metric_kwargs)
    return _make_summary_pandas(scores, columns=pd.Index(["value"]))


def _compute_between_metrics_summaries(
    cluster_preds: list[np.typing.ArrayLike],
    between_metrics: Mapping[str, Callable[..., Any]],
    sample_weight: np.typing.ArrayLike | None = None,
) -> pd.DataFrame:
    """Compute mean and standard deviation of metrics comparing all pairs of clusterings."""
    for i, x in enumerate(cluster_preds):
        cluster_preds[i] = check_array(x, ensure_2d=False)
    metric_kwargs = {}
    if sample_weight is not None:
        sample_weight = check_array(sample_weight, ensure_2d=False)
        check_consistent_length(cluster_preds[0], sample_weight)
        metric_kwargs["sample_weight"] = sample_weight

    scores: dict[str, Any] = {}
    for name, metric in between_metrics.items():
        scores_single = []
        for x, y in itertools.combinations(cluster_preds, r=2):
            res = metric(x, y, **metric_kwargs)
            scores_single.append(res)
        scores[name] = (np.mean(scores_single), np.std(scores_single))
    return _make_summary_pandas(scores, columns=pd.Index(["mean", "std"]))


def bootstrap(
    estimator: BaseEstimator,
    X: np.typing.ArrayLike,
    n_boot: int = 10,
    within_metrics: Mapping[str, Callable[..., Any]] | None = None,
    between_metrics: Mapping[str, Callable[..., Any]] | None = None,
    sample_weight: np.typing.ArrayLike | None = None,
    rng: np.random.Generator | np.random.BitGenerator | None = None,
) -> pd.DataFrame:
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
        within_metrics = {"score": lambda estimator, X, **kwargs: estimator.score(X, **kwargs)}
    if between_metrics is None:
        between_metrics = {"ari": adjusted_rand_score}

    rng = np.random.default_rng(rng)

    # Do bootstrapping and compute within metrics
    within_scores = []
    cluster_preds = []
    for i in range(n_boot):
        estimator_boot = clone(estimator)
        idx_boot = resample(
            np.arange(len(X)), replace=True, n_samples=len(X), random_state=np.random.RandomState(rng.bit_generator)
        )
        X_boot = X[idx_boot]

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight[idx_boot]
        estimator_boot.fit(X_boot, **fit_kwargs)
        within_scores_single = _compute_within_metrics_values(estimator_boot, X, within_metrics, **metric_kwargs)
        within_scores.append(within_scores_single)

        cluster_pred = estimator_boot.predict(X)
        cluster_preds.append(cluster_pred)
    within_summaries = (
        pd.concat(within_scores, ignore_index=True)
        .groupby("metric", observed=False)
        .agg({"mean", "std"})
        .droplevel(0, axis="columns")
        .reset_index()
    )

    # Compare clusterings using between metrics
    between_summaries = _compute_between_metrics_summaries(cluster_preds, between_metrics, **metric_kwargs)

    within_summaries["type"] = "within"
    between_summaries["type"] = "between"
    return pd.concat([within_summaries, between_summaries], ignore_index=True).assign(
        **{"type": lambda x: x["type"].astype(pd.CategoricalDtype(["within", "between"]))}
    )


def bootstrap_grid_search(
    estimator: BaseEstimator,
    X: np.typing.ArrayLike,
    param_grid: dict | Iterable[dict],
    n_boot: int = 10,
    within_metrics: Mapping[str, Callable[..., Any]] | None = None,
    between_metrics: Mapping[str, Callable[..., Any]] | None = None,
    sample_weight: np.typing.ArrayLike | None = None,
    rng: np.random.Generator | np.random.BitGenerator | None = None,
) -> pd.DataFrame:
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
        metrics_summaries = bootstrap(
            estimator_inner,
            X,
            n_boot=n_boot,
            within_metrics=within_metrics,
            between_metrics=between_metrics,
            sample_weight=sample_weight,
            rng=rng,
        )

        metrics_summaries = metrics_summaries.assign(**params)
        results.append(metrics_summaries)
    print(f"{n_comb}/{n_comb}")

    return pd.concat(results, ignore_index=True)
