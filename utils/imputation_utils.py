"""Module with utilities for imputations."""

from __future__ import annotations

import typing
import warnings

import miceforest as mf
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

pd.options.mode.copy_on_write = True


class MICEForest(TransformerMixin, BaseEstimator):
    """
    MICE-forest imputer

    A class to perform random forest MICE imputation with predictive mean matching, using the scikit-learn transformer
    template.
    """

    def __init__(
        self,
        mi_number: int = 1,
        mice_iterations: int = 3,
        *,
        rng: np.random.Generator | np.random.BitGenerator | None = None,
        **kwargs,
    ):
        self.mi_number = mi_number
        self.mice_iterations = mice_iterations
        self.rng = np.random.default_rng(rng)
        self.imputation_kernel = mf.ImputationKernel(data=pd.DataFrame())

        # Filter keyword arguments
        self.kernel_kwargs = {
            key: value for key, value in kwargs.items() if key in mf.ImputationKernel.__init__.__code__.co_varnames
        }
        self.mice_kwargs = {key: value for key, value in kwargs.items() if key not in self.kernel_kwargs.keys()}

    def fit(self, X: pd.DataFrame, y: None = None, **kwargs):
        data = X.reset_index(drop=True)

        # Change PMM strategy for categorical variables to avoid numerical errors. See
        # https://github.com/AnotherSamWilson/miceforest/issues/96#issuecomment-3242644783.
        categorical_vars = data.select_dtypes("category").columns
        pmm_strategies = {var: "shap" if var in categorical_vars else "normal" for var in data.columns}

        # Train imputation kernel
        random_state = np.random.RandomState(self.rng.bit_generator)
        self.imputation_kernel = mf.ImputationKernel(
            data,
            num_datasets=self.mi_number,
            random_state=random_state,
            mean_match_strategy=pmm_strategies,
            mean_match_candidates=5,
            **self.kernel_kwargs,
        )
        # Catch PerformanceWarning from DataFrame fracturing. Might still lead to crash if mice_iterations is set higher
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
            self.imputation_kernel.mice(iterations=self.mice_iterations, **self.mice_kwargs)

        return self

    def transform(self, X: pd.DataFrame, use_results_from_training: bool = False):
        """
        Impute data.

        If the training data is passed into transform and use_results_from_training is set to True, then the imputed
        results from training time are used, which is faster.
        """
        # Save and reset index
        index = X.index.copy()
        data = X.reset_index(drop=True)

        # Flag marking whether to run imputation again or use the imputation results from training time
        run_imputation_again = True
        if use_results_from_training:
            data_train = self.imputation_kernel.working_data.reset_index(drop=True)
            # Check if the input data is the training data
            if data_train.index.equals(data.index) and (data.isna() | (data_train == data)).all().all():
                run_imputation_again = False

        kernel = self.imputation_kernel.impute_new_data(data) if run_imputation_again else self.imputation_kernel

        # Perform single imputation using the kernel
        data_imputed = kernel.complete_data(0, inplace=False)
        data_imputed = typing.cast(pd.DataFrame, data_imputed)

        # Set index and return data
        data_imputed.index = index
        return data_imputed
