"""Module with utilities for bootstrapping."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_array
from sklearn.utils.validation import _check_sample_weight

from utils.metrics_utils import adjusted_rand_score

type MetricValue = int | float | np.integer[Any] | np.floating[Any]


class EstimatorMetric(Protocol):
    def __call__(
        self,
        estimator: BaseEstimator,
        X: NDArray[Any],
        labels: NDArray[Any],
        /,
        *,
        sample_weight: NDArray[np.float64] | None,
    ) -> MetricValue: ...


class AgreementMetric(Protocol):
    def __call__(
        self, labels_0: NDArray[Any], labels_1: NDArray[Any], /, *, sample_weight: NDArray[np.float64] | None
    ) -> MetricValue: ...


Scope = Literal["inbag", "oob", "agreement"]
SummaryRow = tuple[float, float, float, int, Scope]
SUMMARY_COLUMNS = pd.Index(["metric", "mean", "mcse", "std", "n_valid", "scope"])


def _check_metric_value(value: object, name: str, *, allow_nan: bool) -> float:
    """Convert a scalar metric value to a float."""
    value_array = np.asarray(value, dtype=np.float64)
    if value_array.ndim != 0 or np.isinf(value_array) or (allow_nan is False and np.isnan(value_array)):
        qualifier = "a finite scalar or NaN" if allow_nan else "a finite scalar"
        raise ValueError(f"Metric {name!r} must return {qualifier}.")
    return float(value_array)


def _compute_metric_values(
    estimator: BaseEstimator,
    X: NDArray[Any],
    labels: NDArray[Any],
    metrics: Mapping[str, EstimatorMetric],
    sample_weight: NDArray[np.float64] | None,
) -> dict[str, float]:
    """Compute metrics for one fitted estimator."""
    return {
        name: _check_metric_value(metric(estimator, X, labels, sample_weight=sample_weight), name, allow_nan=True)
        for name, metric in metrics.items()
    }


def _compute_agreement_results(
    bootstrap_labels: list[NDArray[Any]],
    agreement_metrics: dict[str, AgreementMetric],
    sample_weight: NDArray[np.float64] | None,
) -> pd.DataFrame:
    """Calculate agreement metrics across bootstrap fits."""
    n_boot = len(bootstrap_labels)
    if len(agreement_metrics) != 0 and n_boot < 4:
        raise ValueError("At least 4 bootstrap iterations are required for unbiased estimation.")

    pair_indices: NDArray[np.intp] = np.asarray(list(itertools.combinations(range(n_boot), r=2)), dtype=np.intp)
    scores: dict[str, NDArray[np.float64]] = {}
    for name, metric in agreement_metrics.items():
        pair_scores: list[float] = []
        for i, j in pair_indices:
            pair_scores.append(
                _check_metric_value(
                    metric(bootstrap_labels[i], bootstrap_labels[j], sample_weight=sample_weight), name, allow_nan=False
                )
            )
        scores[name] = np.asarray(pair_scores, dtype=np.float64)
    return pd.DataFrame.from_dict(scores)


def estimate_order_2_u_stat_variance(
    pair_scores: ArrayLike, pair_indices: ArrayLike, n_boot: int, *, use_nonnegative_amendment: bool = False
) -> float:
    """Estimate variance of order-2 U-statistic, optionally using Wang--Lindsay's non-negative amendment."""
    if n_boot < 4:
        raise ValueError("n_boot must be at least 4 for unbiased estimation.")

    pair_scores: NDArray[np.float64] = check_array(pair_scores, dtype=np.float64, ensure_2d=False)
    pair_indices: NDArray[np.intp] = check_array(pair_indices, dtype=np.intp)
    n_pairs = math.comb(n_boot, 2)
    if pair_scores.shape != (n_pairs,) or pair_indices.shape != (n_pairs, 2):
        raise ValueError("pair_scores and pair_indices must have compatible shapes.")

    # For all models, get sum of all scores that that model was part in
    centered_scores = pair_scores - pair_scores.mean()
    totals_by_model = np.zeros(n_boot, dtype=np.float64)
    np.add.at(totals_by_model, pair_indices[:, 0], centered_scores)  # a[indices[k]] += values[k]
    np.add.at(totals_by_model, pair_indices[:, 1], centered_scores)  # a[indices[k]] += values[k]

    totals_by_model_sum_of_squares = np.sum(np.square(totals_by_model))
    centered_scores_sum_of_squares = np.sum(np.square(centered_scores))

    # Unbiased variance estimator
    q2_minus_q0 = (totals_by_model_sum_of_squares - centered_scores_sum_of_squares) / (6 * math.comb(n_boot, 4))

    if use_nonnegative_amendment:
        # Negatively biased non-negative variance estimator
        q2_minus_q1 = centered_scores_sum_of_squares / (n_pairs * (n_pairs - 1))

        # Non-negative Wang--Lindsay's amendment is max{Q(2) - Q(0), Q(2) - Q(1)}.
        return float(max(q2_minus_q0, q2_minus_q1))

    return float(q2_minus_q0)


def _rescale_sample_weight(sample_weight: NDArray[np.float64], target_sum: float) -> NDArray[np.float64]:
    """Rescale weights to a fixed positive total."""
    weight_sum = float(sample_weight.sum(dtype=np.float64))
    if weight_sum <= 0:
        raise ValueError("Selected sample weights must have a positive sum.")
    return sample_weight * (target_sum / weight_sum)


def _bootstrap_one_candidate(
    estimator: BaseEstimator,
    X: NDArray[Any],
    metrics: dict[Scope, dict[str, Any]],
    sample_weight: NDArray[np.float64] | None,
    boot_indices: NDArray[np.intp],
) -> dict[Scope, pd.DataFrame]:
    if sample_weight is None:
        full_weight_sum = None
    else:
        full_weight_sum = float(sample_weight.sum())
        if full_weight_sum <= 0:
            raise ValueError("sample_weight must have a positive sum.")

    results_inbag: list[dict[str, float]] = []
    results_oob: list[dict[str, float]] = []
    bootstrap_labels: list[NDArray[Any]] = []
    for idx_boot in boot_indices:
        estimator_boot = clone(estimator)
        multiplicities = np.bincount(idx_boot, minlength=len(X))
        mask_inbag = multiplicities > 0
        mask_oob = ~mask_inbag

        # Construct the in-bag sample and in-bag and out-of-bag sample weights
        if sample_weight is None:
            indices_inbag = idx_boot
            sample_weight_inbag = None
            sample_weight_oob = None
        else:
            if full_weight_sum is None:
                raise ValueError("full_weight_sum may not be None if sample_weight is not None.")
            indices_inbag = np.flatnonzero(mask_inbag)
            sample_weight_inbag = _rescale_sample_weight(
                sample_weight[mask_inbag] * multiplicities[mask_inbag], full_weight_sum
            )
            sample_weight_oob = sample_weight[mask_oob]
        X_inbag = X[indices_inbag]

        # Fit estimator
        if sample_weight_inbag is None:
            estimator_boot.fit(X_inbag)
        else:
            estimator_boot.fit(X_inbag, sample_weight=sample_weight_inbag)

        # Predict labels on full data
        labels: NDArray[Any] = np.asarray(estimator_boot.predict(X))
        bootstrap_labels.append(labels)

        # Calculate in-bag metrics
        results_inbag.append(
            _compute_metric_values(
                estimator_boot, X_inbag, labels[indices_inbag], metrics["inbag"], sample_weight_inbag
            )
        )

        # Calculate out-of-bag metrics
        if not np.any(mask_oob) or (sample_weight_oob is not None and sample_weight_oob.sum() <= 0):
            results_oob.append(dict.fromkeys(metrics["oob"], math.nan))
        else:
            results_oob.append(
                _compute_metric_values(estimator_boot, X[mask_oob], labels[mask_oob], metrics["oob"], sample_weight_oob)
            )

    # Calculate agreement metrics and return all metrics
    return {
        "inbag": pd.DataFrame(results_inbag),
        "oob": pd.DataFrame(results_oob),
        "agreement": _compute_agreement_results(bootstrap_labels, metrics["agreement"], sample_weight),
    }


def _summarize_metrics_one_candidate(
    params: dict[str, Any], results: pd.DataFrame, scope: Scope, use_nonnegative_amendment: bool
) -> pd.DataFrame:
    """
    Summarize metric values across bootstrap fits.

    The mean is an order-1 U-statistic for scopes "inbag" and "oob" and an order-2 U-statistic for scope "agreement".
    """
    if scope == "agreement":

        def n_valid(x: pd.Series) -> int:
            return (math.isqrt(8 * x.count() + 1) + 1) // 2

        def mcse(x: pd.Series) -> float:
            pair_scores: NDArray[np.float64] = np.asarray(x.values, dtype=np.float64)
            n_boot = (math.isqrt(8 * len(pair_scores) + 1) + 1) // 2
            pair_indices: NDArray[np.intp] = np.asarray(list(itertools.combinations(range(n_boot), r=2)), dtype=np.intp)
            mc_var = estimate_order_2_u_stat_variance(
                pair_scores, pair_indices, n_boot, use_nonnegative_amendment=use_nonnegative_amendment
            )
            return math.sqrt(mc_var)

        summary = (
            results.aggregate([n_valid, "mean", mcse])
            .set_axis(["n_valid", "mean", "mcse"])
            .transpose()
            .assign(**{"std": lambda x: np.sqrt(x["n_valid"]) * x["mcse"]})
        )
    else:
        summary = (
            results.aggregate(["count", "mean", lambda x: x.std(ddof=1)])
            .set_axis(["n_valid", "mean", "std"])
            .transpose()
            .assign(**{"mcse": lambda x: x["std"] / np.sqrt(x["n_valid"].replace(0, np.nan))})
        )
    return (
        summary.assign(**{"n_valid": lambda x: x["n_valid"].astype("int"), "scope": scope})
        .reset_index(names="metric")
        .reindex(columns=SUMMARY_COLUMNS)
        .assign(**params)
    )


class BootstrapGridSearch:
    """
    Bootstrap models with candidate parameters using shared resamples and select the best ones using metrics.

    In-bag and out-of-bag metrics have signature `(estimator, X, labels, *, sample_weight)`, they only use in-bag or
    out-of-bag data, labels and sample weights respectively. Agreement metrics have signature
    `(labels_0, labels_1, *, sample_weight)` and compare the predictions of all original observations from two models.
    If no weights are supplied, metrics receive `None` for their `sample_weight` argument.

    Each resample contains `len(X)` uniformly drawn observations. In-bag weights combine the bootstrap multiplicities
    with the supplied weights and are rescaled to the original weight total. Out-of-bag metrics use the original
    weights of the held-out observations without rescaling.

    After bootstrapping, eligible parameter combinations are determined using `selection_metric`. A parameter
    combination is eligible if the confidence interval for its difference from the maximum observed value contains zero,
    at the confidence level specified by `selection_confidence`. The best parameter combination is selected from the
    eligible parameter combinations by optimizing `tie_break`. Metrics specified by `selection_metric` or `tie_break`
    are maximized by default. To minimize a metric, its name needs to be prefixed with a minus sign. If `tie_break` is
    `None`, the first occurring eligible parameter combination in the input parameter grid is selected.

    The following fields are filled when the model is fitted:
    - summaries_: Metric summaries for all parameter combinations, with the mean of the metric over bootstrap samples
    and its Monte Carlo standard error, the standard deviation of the metric between bootstrap samples, and the number
    of valid iterations (where the metric was finite).
    - paired_difference_summaries_: The summary of the difference from the maximum observed value, used for finding
    eligible parameter combinations.
    - eligible_params_: The parameter combinations where `selection_metric` is close enough to its maximum observed
    value, in the sense described above.
    - best_params_: The parameter combination selected using the two step selection criterion.
    - best_estimator_: The estimator with the selected parameter combination, refitted on the complete data.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: Mapping[str, Sequence[Any] | NDArray[Any]] | Sequence[Mapping[str, Sequence[Any] | NDArray[Any]]],
        *,
        n_boot: int = 10,
        inbag_metrics: Mapping[str, EstimatorMetric] | None = None,
        oob_metrics: Mapping[str, EstimatorMetric] | None = None,
        agreement_metrics: Mapping[str, AgreementMetric] | None = None,
        selection_metric: str | None = None,
        tie_break: str | None = None,
        selection_confidence: float = 0.95,
        use_nonnegative_amendment: bool = True,
        rng: np.random.Generator | np.random.BitGenerator | None = None,
    ) -> None:
        self.estimator = estimator
        self.param_grid = ParameterGrid(param_grid)
        if len(param_grid) == 0:
            raise ValueError("param_grid must be non-empty.")

        if n_boot < 2:
            raise ValueError("n_boot must be at least 2.")
        self.n_boot = n_boot

        if inbag_metrics is None:
            inbag_metrics = {}
        if oob_metrics is None:
            oob_metrics = {
                "score": lambda estimator, X, labels, sample_weight: (
                    estimator.score(X) if sample_weight is None else estimator.score(X, sample_weight=sample_weight)
                )
            }
        if agreement_metrics is None:
            agreement_metrics = {"ari": adjusted_rand_score}
        self.metrics: dict[Scope, dict[str, Any]] = {
            "inbag": dict(inbag_metrics),
            "oob": dict(oob_metrics),
            "agreement": dict(agreement_metrics),
        }
        if all(len(metrics) == 0 for metrics in self.metrics.values()):
            raise ValueError("At least one metric is required.")
        if len(self.metrics["agreement"]) != 0 and n_boot < 4:
            raise ValueError("At least 4 bootstrap iterations are required for unbiased estimation.")

        if selection_metric is None:
            self.selection_metric = next(iter(next(itertools.chain(self.metrics.values()))))
            self.selection_metric_direction = "max"
        elif selection_metric.lstrip().startswith("-"):
            self.selection_metric = selection_metric.lstrip(" -")
            self.selection_metric_direction = "min"
        else:
            self.selection_metric = selection_metric
            self.selection_metric_direction = "max"
        if tie_break is None:
            self.tie_break = None
            self.tie_break_direction = "min"
        elif tie_break.lstrip().startswith("-"):
            self.tie_break = tie_break.lstrip(" -")
            self.tie_break_direction = "min"
        else:
            self.tie_break = tie_break
            self.tie_break_direction = "max"

        if not 0 <= selection_confidence <= 1:
            raise ValueError("selection_confidence must be in range [0, 1].")
        self.selection_confidence = selection_confidence

        self.use_nonnegative_amendment = use_nonnegative_amendment
        self.rng = np.random.default_rng(rng)

    def fit(
        self, X: ArrayLike, *, sample_weight: ArrayLike | None = None, verbose: bool = False
    ) -> BootstrapGridSearch:
        """Run the bootstrap search and optionally refit the selected estimator."""
        X: NDArray[Any] = check_array(X, force_all_finite=False)
        if sample_weight is not None:
            sample_weight: NDArray[np.float64] = _check_sample_weight(
                sample_weight, X, dtype=np.float64, only_non_negative=True
            )

        boot_indices = self.rng.integers(len(X), size=(self.n_boot, len(X)), dtype=np.intp)
        n_comb = len(self.param_grid)

        selection_metric_scope = next(
            scope for scope, metrics in self.metrics.items() if self.selection_metric in metrics
        )

        summaries: list[pd.DataFrame] = []
        selection_metric_results_list: list[tuple[dict[str, Any], pd.Series]] = []
        for i, params in enumerate(self.param_grid):
            if verbose:
                print(f"parameter combination {i}/{n_comb}", end="\r")

            # Run n_boot bootstrap iterations
            estimator_inner = clone(self.estimator).set_params(**params)
            results = _bootstrap_one_candidate(
                estimator_inner, X, metrics=self.metrics, sample_weight=sample_weight, boot_indices=boot_indices
            )

            # Store the results of all iterations for the selection metric
            selection_metric_results: pd.Series = results[selection_metric_scope][self.selection_metric].pipe(
                lambda x: -x if self.selection_metric_direction == "min" else x
            )
            selection_metric_results_list.append((params, selection_metric_results))

            # Store only summaries for all other metrics
            summaries.extend(
                _summarize_metrics_one_candidate(params, results[scope], scope, self.use_nonnegative_amendment)
                for scope, metrics in self.metrics.items()
                if len(metrics) != 0
            )
        if verbose:
            print(f"parameter combination {n_comb}/{n_comb}")

        # Store summaries
        if len(summaries) == 0:
            self.summaries_ = pd.DataFrame.from_dict(
                {}, orient="index", columns=pd.Index([*SUMMARY_COLUMNS, *list(self.param_grid[0])])
            )
        else:
            self.summaries_ = pd.concat(summaries, ignore_index=True)
        self.summaries_ = self.summaries_.assign(
            **{"scope": lambda x: x["scope"].astype(pd.CategoricalDtype(["inbag", "oob", "agreement"]))}
        )

        # Get eligible parameters, where 0 in [mean_paired_diffs +- quant(selection_confidence) * std_mean_paired_diffs]
        max_params = (
            self.summaries_[lambda x: x["metric"].eq(self.selection_metric)]
            .pipe(
                lambda x: x.assign(**{"mean": lambda x: -x["mean"]}) if self.selection_metric_direction == "min" else x
            )
            .nlargest(1, "mean")
            .drop(columns=SUMMARY_COLUMNS)
            .to_dict(orient="records")[0]
        )
        max_results = next(results for params, results in selection_metric_results_list if params == max_params)
        summaries_paired_diffs: list[pd.DataFrame] = []
        for params, results in selection_metric_results_list:
            summaries_paired_diffs.append(
                _summarize_metrics_one_candidate(
                    params,
                    pd.DataFrame({"paired_diff": max_results - results}),
                    selection_metric_scope,
                    self.use_nonnegative_amendment,
                )
            )
        self.paired_difference_summaries_ = pd.concat(summaries_paired_diffs, ignore_index=True).assign(
            **{"lower_bound": lambda x: x["mean"] - norm.ppf(self.selection_confidence) * x["mcse"]}
        )
        self.eligible_params_ = (
            self.paired_difference_summaries_[lambda x: x["lower_bound"] <= 0]
            .drop(columns=pd.Index([*SUMMARY_COLUMNS, "lower_bound"]))
            .to_dict(orient="records")
        )
        if verbose:
            print(f"eligible parameters: {self.eligible_params_}")

        # Get best parameters by optimizing tie_break among eligible parameters
        if self.tie_break is None:
            self.best_params_ = min(self.eligible_params_, key=lambda x: tuple(x.values()))
        else:
            self.best_params_ = (
                self.summaries_.merge(pd.DataFrame(self.eligible_params_))
                .loc[lambda x: x["metric"].eq(self.tie_break)]
                .pipe(lambda x: x.assign(**{"mean": lambda x: -x["mean"]}) if self.tie_break_direction == "min" else x)
                .loc[lambda x: x["mean"].idxmax()]
                .drop(index=SUMMARY_COLUMNS)
                .to_dict()
            )
        if verbose:
            print(f"best parameters: {self.best_params_}")

        # Refit model on full data
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
        if sample_weight is None:
            self.best_estimator_ = self.best_estimator_.fit(X)
        else:
            self.best_estimator_ = self.best_estimator_.fit(X, sample_weight=sample_weight)

        return self
