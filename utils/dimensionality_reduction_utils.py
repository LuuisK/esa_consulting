"""Module with utilities for dimensionality reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

pd.options.mode.copy_on_write = True


class BlockPCA(TransformerMixin, BaseEstimator):
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
    ):
        self.var_groups = var_groups
        if isinstance(n_components_per_group, int):
            self.n_components_per_group = {group: n_components_per_group for group in var_groups.keys()}
        else:
            self.n_components_per_group = n_components_per_group
        self.models = {group: PCA() for group in var_groups.keys()}
        self.rng = np.random.default_rng(rng)
        self.pca_kwargs = kwargs

    def fit(self, X: pd.DataFrame, y: None = None, **kwargs):
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

        return self

    def transform(self, X: pd.DataFrame):
        data = X

        # Save index
        index = data.index.copy()

        data_pca: list[pd.DataFrame] = []
        for group, group_vars in self.var_groups.items():
            # Restrict to one group
            model = self.models[group]
            group_data = data[group_vars]

            # Transform
            data_loadings = pd.DataFrame(model.transform(X=group_data)).rename(columns=lambda x: f"{group}_comp_{x}")

            # Save data
            data_pca.append(data_loadings)

        # Concatenate components from groups
        data_pca_merged = pd.concat(data_pca, axis=1)

        # Set index and return data
        data_pca_merged.index = index
        return data_pca_merged

    def inverse_transform(self, X: pd.DataFrame):
        data = X

        # Save index
        index = data.index.copy()

        data_pca_inv: list[pd.DataFrame] = []
        for group, group_vars in self.var_groups.items():
            # Restrict to one group
            model = self.models[group]
            group_data = data.loc[:, data.columns.str.startswith(group)]

            # Transform back
            data_values = pd.DataFrame(model.inverse_transform(X=group_data)).rename(
                columns=lambda x, iter=iter(group_vars): next(iter)
            )

            # Save data
            data_pca_inv.append(data_values)

        # Concatenate components from groups
        data_pca_inv_merged = pd.concat(data_pca_inv, axis=1)

        # Set index and return data
        data_pca_inv_merged.index = index
        return data_pca_inv_merged
