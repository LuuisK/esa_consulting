"""Module with utilities for Gaussian mixture modeling."""

from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from time import time

import numpy as np
from scipy.special import logsumexp
from sklearn import cluster
from sklearn.base import BaseEstimator, DensityMixin, _fit_context
from sklearn.cluster import kmeans_plusplus
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture._gaussian_mixture import (
    _check_means,
    _check_precisions,
    _check_weights,
    _compute_precision_cholesky,
    _compute_precision_cholesky_from_precisions,
    _estimate_gaussian_covariances_diag,
    _estimate_gaussian_covariances_full,
    _estimate_gaussian_covariances_spherical,
    _estimate_log_gaussian_prob,
)
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import _check_sample_weight, check_is_fitted


# Copied from sklearn and modified the following methods to enable sample weights:
# _initialize_parameters, fit, fit_predict, _e_step, and score
# ruff: disable[RUF012, PIE790, B028, UP031, B905]
class BaseMixture(DensityMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for mixture models.

    This abstract class specifies an interface for all mixture classes and
    provides basic common methods for mixture models.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "reg_covar": [Interval(Real, 0.0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "init_params": [StrOptions({"kmeans", "random", "random_from_data", "k-means++"})],
        "random_state": ["random_state"],
        "warm_start": ["boolean"],
        "verbose": ["verbose"],
        "verbose_interval": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        n_components,
        tol,
        reg_covar,
        max_iter,
        n_init,
        init_params,
        random_state,
        warm_start,
        verbose,
        verbose_interval,
    ):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    @abstractmethod
    def _check_parameters(self, X):
        """Check initial parameters of the derived class.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)
        """
        pass

    def _initialize_parameters(self, X, random_state, sample_weight):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in `X`. If `None`, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.
        """
        n_samples, _ = X.shape

        if self.init_params == "kmeans":
            resp = np.zeros((n_samples, self.n_components))
            label = (
                cluster.KMeans(n_clusters=self.n_components, n_init=1, random_state=random_state)
                .fit(X, sample_weight=sample_weight)
                .labels_
            )
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == "random":
            resp = random_state.uniform(size=(n_samples, self.n_components))
            resp /= resp.sum(axis=1)[:, np.newaxis]
        elif self.init_params == "random_from_data":
            resp = np.zeros((n_samples, self.n_components))
            probabilities = None if sample_weight is None else sample_weight / sample_weight.sum(dtype=np.float64)
            indices = random_state.choice(n_samples, p=probabilities, size=self.n_components, replace=False)
            resp[indices, np.arange(self.n_components)] = 1
        elif self.init_params == "k-means++":
            resp = np.zeros((n_samples, self.n_components))
            _, indices = kmeans_plusplus(
                X,
                self.n_components,
                sample_weight=sample_weight,
                random_state=random_state,
            )
            resp[indices, np.arange(self.n_components)] = 1

        self._initialize(X, resp, sample_weight)

    @abstractmethod
    def _initialize(self, X, resp, sample_weight):
        """Initialize the model parameters of the derived class.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in `X`. If `None`, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.
        """
        pass

    def fit(self, X, y=None, sample_weight=None):
        """Estimate model parameters with the EM algorithm.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
        initialization is performed upon the first call. Upon consecutive
        calls, training starts where it left off.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in `X`. If `None`, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

        Returns
        -------
        self : object
            The fitted mixture.
        """
        # parameters are validated in fit_predict
        self.fit_predict(X, y, sample_weight=sample_weight)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_predict(self, X, y=None, sample_weight=None):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in `X`. If `None`, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_parameters(X)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype, only_non_negative=True)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state, sample_weight)

            lower_bound = -np.inf if do_init else self.lower_bound_

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                converged = False
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound

                    log_prob_norm, log_resp = self._e_step(X, sample_weight)
                    self._m_step(X, log_resp, sample_weight)
                    lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

                    change = lower_bound - prev_lower_bound
                    self._print_verbose_msg_iter_end(n_iter, change)

                    if abs(change) < self.tol:
                        converged = True
                        break

                self._print_verbose_msg_init_end(lower_bound, converged)

                if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_n_iter = n_iter
                    self.converged_ = converged

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        if not self.converged_ and self.max_iter > 0:
            warnings.warn(
                (
                    "Best performing initialization did not converge. "
                    "Try different init parameters, or increase max_iter, "
                    "tol, or check for degenerate data."
                ),
                ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X, sample_weight)

        return log_resp.argmax(axis=1)

    def _e_step(self, X, sample_weight):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in `X`. If `None`, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.average(log_prob_norm, weights=sample_weight), log_resp

    @abstractmethod
    def _m_step(self, X, log_resp, sample_weight):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in `X`. If `None`, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.
        """
        pass

    @abstractmethod
    def _get_parameters(self):
        pass

    @abstractmethod
    def _set_parameters(self, params):
        pass

    def score_samples(self, X):
        """Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample in `X` under the current model.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)

    def score(self, X, y=None, sample_weight=None):
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in `X`. If `None`, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of `X` under the Gaussian mixture model.
        """
        scores = self.score_samples(X)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, scores, dtype=np.float64, only_non_negative=True)
        return np.average(scores, weights=sample_weight)

    def predict(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def predict_proba(self, X):
        """Evaluate the components' density for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Density of each Gaussian component for each sample in X.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        _, log_resp = self._estimate_log_prob_resp(X)
        return np.exp(log_resp)

    def sample(self, n_samples=1):
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.

        y : array, shape (nsamples,)
            Component labels.
        """
        check_is_fitted(self)

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at least one sample." % (self.n_components)
            )

        _, n_features = self.means_.shape
        rng = check_random_state(self.random_state)
        n_samples_comp = rng.multinomial(n_samples, self.weights_)

        if self.covariance_type == "full":
            X = np.vstack(
                [
                    rng.multivariate_normal(mean, covariance, int(sample))
                    for (mean, covariance, sample) in zip(self.means_, self.covariances_, n_samples_comp)
                ]
            )
        elif self.covariance_type == "tied":
            X = np.vstack(
                [
                    rng.multivariate_normal(mean, self.covariances_, int(sample))
                    for (mean, sample) in zip(self.means_, n_samples_comp)
                ]
            )
        else:
            X = np.vstack(
                [
                    mean + rng.standard_normal(size=(sample, n_features)) * np.sqrt(covariance)
                    for (mean, covariance, sample) in zip(self.means_, self.covariances_, n_samples_comp)
                ]
            )

        y = np.concatenate([np.full(sample, j, dtype=int) for j, sample in enumerate(n_samples_comp)])

        return (X, y)

    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    @abstractmethod
    def _estimate_log_weights(self):
        """Estimate log-weights in EM algorithm, E[ log pi ] in VB algorithm.

        Returns
        -------
        log_weight : array, shape (n_components, )
        """
        pass

    @abstractmethod
    def _estimate_log_prob(self, X):
        """Estimate the log-probabilities log P(X | Z).

        Compute the log-probabilities per each component for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
        """
        pass

    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _print_verbose_msg_init_beg(self, n_init):
        """Print verbose message on initialization."""
        if self.verbose == 1:
            print("Initialization %d" % n_init)
        elif self.verbose >= 2:
            print("Initialization %d" % n_init)
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time

    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("  Iteration %d" % n_iter)
            elif self.verbose >= 2:
                cur_time = time()
                print(
                    "  Iteration %d\t time lapse %.5fs\t ll change %.5f"
                    % (n_iter, cur_time - self._iter_prev_time, diff_ll)
                )
                self._iter_prev_time = cur_time

    def _print_verbose_msg_init_end(self, lb, init_has_converged):
        """Print verbose message on the end of iteration."""
        converged_msg = "converged" if init_has_converged else "did not converge"
        if self.verbose == 1:
            print(f"Initialization {converged_msg}.")
        elif self.verbose >= 2:
            t = time() - self._init_prev_time
            print(f"Initialization {converged_msg}. time lapse {t:.5f}s\t lower bound {lb:.5f}.")


# Copied from sklearn and modified to enable sample weights
def _estimate_gaussian_covariances_tied(resp, X, nk, means, reg_covar):
    """Estimate the tied covariance matrix.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariance : array, shape (n_features, n_features)
        The tied covariance matrix of the components.
    """
    # In the unweighted case these are 1, but they need to be added here
    sample_weight = resp.sum(axis=1)

    avg_X2 = np.dot((X * sample_weight[:, None]).T, X)
    avg_means2 = np.dot(nk * means.T, means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance.flat[:: len(covariance) + 1] += reg_covar
    return covariance


# Copied from sklearn and modified to enable sample weights
def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type, sample_weight):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.

    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    sample_weight : array-like of shape (n_samples,), default=None
        The weights for each observation in `X`. If `None`, all observations
        are assigned equal weight. `sample_weight` is not used during
        initialization if `init` is a callable or a user provided array.

    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    if sample_weight is not None:
        # Responsibilities always appear multiplied by weights
        resp *= sample_weight[:, None]

    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        "tied": _estimate_gaussian_covariances_tied,
        "diag": _estimate_gaussian_covariances_diag,
        "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, means, covariances


# ruff: enable[RUF012, PIE790, B028, UP031, B905]


# Copied from sklearn and modified the following methods to enable sample weights:
# _initialize_parameters, _initialize, _m_step, bic, and aic
# ruff: disable[RUF012]
class GaussianMixture(BaseMixture):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        String describing the type of covariance parameters to use.
        Must be one of:

        - 'full': each component has its own general covariance matrix.
        - 'tied': all components share the same general covariance matrix.
        - 'diag': each component has its own diagonal covariance matrix.
        - 'spherical': each component has its own single variance.

    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, default=100
        The number of EM iterations to perform.

    n_init : int, default=1
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'k-means++', 'random', 'random_from_data'}, \
    default='kmeans'
        The method used to initialize the weights, the means and the
        precisions.
        String must be one of:

        - 'kmeans' : responsibilities are initialized using kmeans.
        - 'k-means++' : use the k-means++ method to initialize.
        - 'random' : responsibilities are initialized randomly.
        - 'random_from_data' : initial means are randomly selected data points.

        .. versionchanged:: v1.1
            `init_params` now accepts 'random_from_data' and 'k-means++' as
            initialization methods.

    weights_init : array-like of shape (n_components, ), default=None
        The user-provided initial weights.
        If it is None, weights are initialized using the `init_params` method.

    means_init : array-like of shape (n_components, n_features), default=None
        The user-provided initial means,
        If it is None, means are initialized using the `init_params` method.

    precisions_init : array-like, default=None
        The user-provided initial precisions (inverse of the covariance
        matrices).
        If it is None, precisions are initialized using the 'init_params'
        method.
        The shape depends on 'covariance_type'::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters (see `init_params`).
        In addition, it controls the generation of random samples from the
        fitted distribution (see the method `sample`).
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        In that case, 'n_init' is ignored and only a single initialization
        occurs upon the first call.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default=10
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like of shape (n_components,)
        The weights of each mixture components.

    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence of the best fit of EM was reached, False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best fit of EM.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    BayesianGaussianMixture : Gaussian mixture model fit with a variational
        inference.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.mixture import GaussianMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    >>> gm.means_
    array([[10.,  2.],
           [ 1.,  2.]])
    >>> gm.predict([[0, 0], [12, 3]])
    array([1, 0])
    """

    _parameter_constraints: dict = {
        **BaseMixture._parameter_constraints,
        "covariance_type": [StrOptions({"full", "tied", "diag", "spherical"})],
        "weights_init": ["array-like", None],
        "means_init": ["array-like", None],
        "precisions_init": ["array-like", None],
    }

    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )

        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        _, n_features = X.shape

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init, self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init, self.n_components, n_features)

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(
                self.precisions_init,
                self.covariance_type,
                self.n_components,
                n_features,
            )

    def _initialize_parameters(self, X, random_state, sample_weight):
        # If all the initial parameters are all provided, then there is no need to run
        # the initialization.
        compute_resp = self.weights_init is None or self.means_init is None or self.precisions_init is None
        if compute_resp:
            super()._initialize_parameters(X, random_state, sample_weight)
        else:
            self._initialize(X, None, sample_weight)

    def _initialize(self, X, resp, sample_weight):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in `X`. If `None`, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.
        """
        n_samples, _ = X.shape
        weights, means, covariances = None, None, None
        if resp is not None:
            weights, means, covariances = _estimate_gaussian_parameters(
                X, resp, self.reg_covar, self.covariance_type, sample_weight
            )
            if self.weights_init is None:
                n_eff = n_samples if sample_weight is None else sample_weight.sum()
                weights /= n_eff

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(covariances, self.covariance_type)
        else:
            self.precisions_cholesky_ = _compute_precision_cholesky_from_precisions(
                self.precisions_init, self.covariance_type
            )

    def _m_step(self, X, log_resp, sample_weight):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in `X`. If `None`, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.
        """
        resp = np.exp(log_resp)
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type, sample_weight
        )
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, self.covariance_type)

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(X, self.means_, self.precisions_cholesky_, self.covariance_type)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _get_parameters(self):
        return (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        )

    def _set_parameters(self, params):
        (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        ) = params

        # Attributes computation
        _, n_features = self.means_.shape

        if self.covariance_type == "full":
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

        elif self.covariance_type == "tied":
            self.precisions_ = np.dot(self.precisions_cholesky_, self.precisions_cholesky_.T)
        else:
            self.precisions_ = self.precisions_cholesky_**2

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.means_.shape
        if self.covariance_type == "full":
            cov_params = self.n_components * n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "diag":
            cov_params = self.n_components * n_features
        elif self.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "spherical":
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X, sample_weight=None):
        """Bayesian information criterion for the current model on the input X.

        You can refer to this :ref:`mathematical section <aic_bic>` for more
        details regarding the formulation of the BIC used.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in `X`. If `None`, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

        Returns
        -------
        bic : float
            The lower the better.
        """
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64, only_non_negative=True)
        n_eff = X.shape[0] if sample_weight is None else sample_weight.sum()
        return -2 * self.score(X, sample_weight=sample_weight) * n_eff + self._n_parameters() * np.log(n_eff)

    def icl(self, X, sample_weight=None):
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64, only_non_negative=True)
        bic = self.bic(X, sample_weight=sample_weight)

        _, log_resp = self._estimate_log_prob_resp(X)
        resp = np.exp(log_resp)

        if sample_weight is not None:
            resp *= sample_weight[:, None]

        entropy = -np.sum(resp * log_resp)
        return bic + 2 * entropy

    def aic(self, X, sample_weight=None):
        """Akaike information criterion for the current model on the input X.

        You can refer to this :ref:`mathematical section <aic_bic>` for more
        details regarding the formulation of the AIC used.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in `X`. If `None`, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

        Returns
        -------
        aic : float
            The lower the better.
        """
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64, only_non_negative=True)
        n_eff = X.shape[0] if sample_weight is None else sample_weight.sum()
        return -2 * self.score(X, sample_weight=sample_weight) * n_eff + 2 * self._n_parameters()


# ruff: enable[RUF012]
