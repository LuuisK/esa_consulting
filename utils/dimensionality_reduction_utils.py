"""Module with utilities for dimensionality reduction."""

from __future__ import annotations

import itertools
from typing import Any, Self

import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.decomposition import PCA
from sklearn.decomposition._base import _BasePCA

pd.options.mode.copy_on_write = True


class BlockPCA(_BasePCA):
    """
    Block PCA

    A class to perform PCA on variable groups, using the scikit-learn transformer template.
    """

    def __init__(
        self,
        var_groups: dict[str, list[str]],
        n_components_per_group: int | dict[str, int],
        *,
        rng: np.random.Generator | np.random.BitGenerator | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.var_groups = var_groups
        if isinstance(n_components_per_group, int):
            self.n_components_per_group = {group: n_components_per_group for group in var_groups}
        else:
            self.n_components_per_group = n_components_per_group
        self.models = {group: PCA() for group in var_groups}
        self.rng = np.random.default_rng(rng)
        self.pca_kwargs = kwargs
        self.whiten = False

    def fit(self, X: pd.DataFrame, y: None = None, **kwargs) -> Self:
        """
        Fit BlockPCA model.

        The column names of X have to be made up of the names
        ("{group}_comp_{i}" for group in self.var_groups for i in self.n_components_per_group[group]). Input y is
        ignored. Input keyword arguments are passed to all PCA constructors.
        """

        # Set feature_names_in_
        self._check_feature_names(X, reset=True)

        data = X

        for group, group_vars in self.var_groups.items():
            # Restrict to one group
            group_data = data[group_vars]
            n_components = self.n_components_per_group[group]

            # Fit model
            random_state = np.random.RandomState(self.rng.bit_generator)
            kwargs = self.pca_kwargs | kwargs
            model = PCA(n_components=n_components, random_state=random_state, **kwargs)
            model.fit(group_data)

            # Save model
            self.models[group] = model

        # Build mean vector and component matrix
        self.mean_ = np.concat([model.mean_ for model in self.models.values()])
        self.components_ = scipy.linalg.block_diag(*(model.components_ for model in self.models.values()))
        # Assign empty array to explained_variance_ so that super().transform(...) works
        self.explained_variance_ = np.array([])
        # # Assign empty array to noise_variance_
        # self.noise_variance_ = np.array([])

        return self

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(
            [f"{group}_comp_{i}" for group in self.var_groups for i in range(self.n_components_per_group[group])]
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        BlockPCA transform.

        Can only be applied if the model has been fitted. The column names of X have to be made up of the column names
        of the data that this model was fitted on.
        """
        data = X

        # Save index
        index = data.index.copy()

        vars_in_order = itertools.chain.from_iterable(self.var_groups.values())
        columns_in_order = (
            f"{group}_comp_{i}" for group in self.var_groups for i in range(self.n_components_per_group[group])
        )
        data_pca = pd.DataFrame(super().transform(data[vars_in_order].to_numpy())).rename(
            columns=lambda x: next(columns_in_order)
        )

        # Set index and return data
        data_pca.index = index
        return data_pca

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse BlockPCA transform.

        Can only be applied if the model has been fitted. The column names of X have to be made up of the names
        ("{group}_comp_{i}" for group in self.var_groups for i in range(self.n_components_per_group[group])).
        """
        data = X

        # Save index and columns
        index = data.index.copy()

        vars_in_order = itertools.chain.from_iterable(self.var_groups.values())
        columns_in_order = (
            f"{group}_comp_{i}" for group in self.var_groups for i in range(self.n_components_per_group[group])
        )
        data_pca_inv = pd.DataFrame(super().inverse_transform(data[columns_in_order].to_numpy())).rename(
            columns=lambda x: next(vars_in_order)
        )

        # Set index and columns and return data
        data_pca_inv.index = index
        return data_pca_inv
