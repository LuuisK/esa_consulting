"""Module with miscellaneous utilities."""

from __future__ import annotations

import typing
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from umap import UMAP

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

# Create projection method types, lists, and
DeterministicProjection = Isomap
deterministic_projection_list = [Isomap]
str_to_deterministic_projection: dict[str, type[DeterministicProjection]] = {
    x.__name__.lower(): x for x in deterministic_projection_list
}

RandomProjection = LocallyLinearEmbedding | MDS | SpectralEmbedding | TSNE | UMAP
random_projection_list = [LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE, UMAP]
str_to_random_projection: dict[str, type[RandomProjection]] = {x.__name__.lower(): x for x in random_projection_list}

Projection = DeterministicProjection | RandomProjection
projection_list = [*deterministic_projection_list, *random_projection_list]
str_to_projection: dict[str, type[Projection]] = {x.__name__.lower(): x for x in projection_list}


def reduce_dimen_2d(data: pd.DataFrame, projection_method: str | type[Projection] | None = None, **kwargs):
    """Projects data to 2d.

    The data is projected with the method specified as projection_method, if left unspecified the first two coordinates
    of the data are taken as the 2d representation. Valid projection methods are Isomap, LocallyLinearEmbedding, MDS,
    SpectralEmbedding, and TSNE. Certain projection methods require random numbers, otherwise a given RNG is ignored.
    """
    if projection_method is None:
        model_type = None
    elif isinstance(projection_method, str):
        # Get model from string
        model_type = str_to_projection[projection_method.lower()]
    elif issubclass(projection_method, Projection):
        model_type = projection_method

    if model_type is None:
        # Reduce to first two variables
        data_2d = data[:, :2].to_numpy()
    else:
        rng = np.random.default_rng(kwargs.pop("rng", None))
        if issubclass(model_type, RandomProjection):
            # Set random state in keyword argument dictionary
            kwargs["random_state"] = np.random.RandomState(rng.bit_generator)
        # Reduce dimensionality
        transform_2d = model_type(n_components=2, n_jobs=-1, **kwargs)
        data_2d = transform_2d.fit_transform(data)
        data_2d = typing.cast(np.typing.NDArray, data_2d)

    return pd.DataFrame(data={"x": data_2d[:, 0], "y": data_2d[:, 1]}, index=data.index.copy())


def turn_cats_to_continuous(data: pd.DataFrame, treat_cat_as_continuous: Iterable[str] | None) -> pd.DataFrame:
    """
    Turn specified variables of data into continuous ones.

    Used for categorical (pandas.CategoricalDType) variables that should be treated as continuous.
    """
    # Return a modified copy of the input
    data = data.copy()

    treat_cat_as_continuous = [] if treat_cat_as_continuous is None else list(treat_cat_as_continuous)
    treat_cat_as_continuous = [var for var in treat_cat_as_continuous if var in data.columns]
    data[treat_cat_as_continuous] = data[treat_cat_as_continuous].apply(lambda col: col.cat.codes.replace(-1, np.nan))
    return data


def stack_dataframes(dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Turn dataframes with equal index and column labels into one dataframe with two column index levels."""
    all_vars = next(iter(dataframes.values())).columns
    return pd.concat(dataframes, axis=1).swaplevel(0, 1, axis=1).reindex(all_vars, axis=1, level=0)


def contingency_table(index: pd.Series, columns: pd.Series, dropna: bool = False, **kwargs) -> pd.DataFrame:
    """
    Contingency table.

    pd.crosstab that does not ignore NaN.
    """
    assert pd.options.future.infer_string

    index = index.astype("str").fillna("NaN")
    columns = columns.astype("str").fillna("NaN")

    crosstab = pd.crosstab(index, columns, dropna=False, **kwargs)
    crosstab = crosstab.loc[
        [x for x in crosstab.index if x != "NaN"] + ["NaN"], [x for x in crosstab.columns if x != "NaN"] + ["NaN"]
    ]

    if dropna:
        crosstab = crosstab.drop(index="NaN", columns="NaN")

    return crosstab
