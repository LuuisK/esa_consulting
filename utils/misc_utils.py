"""Module with miscellaneous utilities."""

from __future__ import annotations

import typing
from collections.abc import Callable, Hashable, Iterable
from typing import Any, Literal, cast, overload

import numpy as np
import pandas as pd
import scipy.sparse as sp
from IPython.display import HTML, display
from numpy.lib._function_base_impl import _weights_are_valid  # type: ignore
from numpy.lib.array_utils import normalize_axis_tuple
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

    return pd.DataFrame(data={"x": data_2d[:, 0], "y": data_2d[:, 1]})


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


def make_nan_a_value(x: pd.Series) -> pd.Series:
    """Make NaN into a value (convert to string dtype if necessary)."""
    if x.dtype == "category":
        x = x.cat.add_categories(["NaN"]).fillna("NaN")
    else:
        x = x.astype("str").fillna("NaN")
    return x


def contingency_table(index: pd.Series, columns: pd.Series, dropna: bool = False, **kwargs) -> pd.DataFrame:
    """
    Contingency table.

    pd.crosstab that does not ignore NaN.
    """
    assert pd.options.future.infer_string

    index = make_nan_a_value(index)
    columns = make_nan_a_value(columns)

    crosstab = pd.crosstab(index, columns, dropna=False, **kwargs)

    if dropna:
        index_nan = crosstab.index[crosstab.index.str.fullmatch("NaN")]
        columns_nan = crosstab.columns[crosstab.columns.str.fullmatch("NaN")]

        crosstab = crosstab.drop(index=index_nan, columns=columns_nan)

    return crosstab


# Largely copied from numpy.average
def var_weighted(a, axis=None, weights=None, returned=False, *, keepdims=np._NoValue):
    a = np.asanyarray(a)

    if axis is not None:
        axis = normalize_axis_tuple(axis, a.ndim, argname="axis")

    if keepdims is np._NoValue:
        # Don't pass on the keepdims argument if one wasn't given.
        keepdims_kw = {}
    else:
        keepdims_kw = {"keepdims": keepdims}

    if weights is None:
        var = a.var(axis, **keepdims_kw)
        var_as_array = np.asanyarray(var)
        scl = var_as_array.dtype.type(a.size / var_as_array.size)
    else:
        wgt = _weights_are_valid(weights=weights, a=a, axis=axis)

        if issubclass(a.dtype.type, (np.integer, np.bool)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, "f8")
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)

        scl = wgt.sum(axis=axis, dtype=result_dtype, **keepdims_kw)
        if np.any(scl == 0.0):
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")

        avg = np.multiply(a, wgt, dtype=result_dtype).sum(axis, keepdims=True) / scl
        sq_diff = np.square(np.subtract(a, avg))
        var = var_as_array = np.multiply(sq_diff, wgt, dtype=result_dtype).sum(axis, **keepdims_kw) / scl

    if returned:
        if scl.shape != var_as_array.shape:
            scl = np.broadcast_to(scl, var_as_array.shape).copy()
        return var, scl
    else:
        return var


def std_weighted(a, axis=None, weights=None, returned=False, *, keepdims=np._NoValue):
    return np.sqrt(var_weighted(a, axis=axis, weights=weights, returned=returned, keepdims=keepdims))


# Copied from sklearn and modified to enable sample weights
def contingency_matrix(labels_true, labels_pred, *, sample_weight=None, eps=None, sparse=False, dtype=None):
    """Build a contingency matrix describing the relationship between labels."""
    if dtype is None:
        dtype = np.int64 if sample_weight is None else np.float64
    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    sample_weight = np.ones(class_idx.shape[0]) if sample_weight is None else sample_weight
    contingency = sp.coo_matrix(
        (sample_weight, (class_idx, cluster_idx)),
        shape=(n_classes, n_clusters),
        dtype=dtype,
    )
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


def value_counts(
    values: pd.Series,
    sort: bool = True,
    ascending: bool = False,
    normalize: bool = False,
    dropna: bool = True,
    sample_weight: pd.Series | np.typing.ArrayLike | None = None,
) -> pd.Series:
    if sample_weight is None:
        frequencies = values.value_counts(normalize=normalize, sort=sort, ascending=ascending, dropna=dropna)
    else:
        frequencies = pd.Series(sample_weight).groupby(values, dropna=dropna, observed=False).sum()
        frequencies.name = "count"
        if normalize:
            frequencies = frequencies / frequencies.sum()
            frequencies.name = "proportion"
        if sort:
            frequencies = frequencies.sort_values(ascending=ascending)
    return frequencies


# See answers to https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
def display_side_by_side(*dfs: pd.DataFrame, captions: str | Iterable[str] | None = None, tablespacing: int = 5):
    if captions is None:
        captions = ""
    if isinstance(captions, str):
        captions = [captions] * len(dfs)
    output = ""
    for df, caption in zip(dfs, captions, strict=True):
        output += (
            df.style.set_caption(caption)
            .set_table_styles([{"selector": "caption", "props": [("font-size", "16px"), ("font-weight", "bold")]}])
            .set_table_attributes("style='display:inline; margin-right:10px;'")
            ._repr_html_()
        ) or ""
        output += tablespacing * "\xa0"
    display(HTML(output))


def _is_column_label(data: pd.DataFrame, x: object) -> bool:
    try:
        return x in data.columns
    except TypeError:
        return False


def _check_agg_with_overall_input(
    data: pd.DataFrame, overall_var: Hashable, by_vars: Hashable | Iterable[Hashable] | None
) -> tuple[pd.DataFrame, Hashable, list[Hashable]]:
    # Check overall_var is valid
    if overall_var not in data.columns:
        raise KeyError(f"{overall_var!r} is not a column in `data`.")

    # Cast by_vars to list
    if by_vars is None:
        by_vars = [overall_var]
    elif _is_column_label(data, by_vars):
        by_vars = [by_vars]
    else:
        by_vars = cast(Iterable[Hashable], by_vars)
        by_vars = list(by_vars)

    # Add overall_var if not present in by_vars
    if overall_var not in by_vars:
        by_vars = [overall_var, *by_vars]

    return data, overall_var, by_vars


def agg_with_overall(
    data: pd.DataFrame,
    overall_var: Hashable,
    *agg_args: Any,
    by_vars: Hashable | Iterable[Hashable] | None = None,
    overall_label: Hashable = "all",
    observed: bool = False,
    **agg_kwargs: Any,
) -> pd.DataFrame:
    """Aggregate by groups and add an overall group.

    `overall_var` is the grouping variable whose value is replaced by `overall_label` for the overall rows.
    """
    data, overall_var, by_vars = _check_agg_with_overall_input(data, overall_var, by_vars)

    data_grouped = data.copy()

    if (
        isinstance(data_grouped[overall_var].dtype, pd.CategoricalDtype)
        and overall_label not in data_grouped[overall_var].cat.categories
    ):
        data_grouped[overall_var] = data_grouped[overall_var].cat.add_categories(overall_label)

    data_overall = data_grouped.assign(
        **{overall_var: lambda x: pd.Series(overall_label, index=x.index, dtype=x[overall_var].dtype)}  # type: ignore
    )

    return (
        pd.concat([data_grouped, data_overall], ignore_index=False)
        .groupby(by_vars, observed=observed)
        .agg(*agg_args, **agg_kwargs)
    )


@overload
def partial_fn_with_info[R](
    fn: Callable[..., R],
    /,
    ignore_nan: bool = False,
    empty_value: float = np.nan,
    *args: pd.Series | None,
    **kwargs: pd.Series | None,
) -> Callable[[pd.Series], R | float]: ...
@overload
def partial_fn_with_info[R, E](
    fn: Callable[..., R],
    /,
    ignore_nan: bool,
    empty_value: E = ...,
    *args: pd.Series | None,
    **kwargs: pd.Series | None,
) -> Callable[[pd.Series], R | E]: ...
def partial_fn_with_info[R, E](
    fn: Callable[..., R],
    /,
    ignore_nan: bool = False,
    empty_value: E | float = np.nan,
    *args: pd.Series | None,
    **kwargs: pd.Series | None,
) -> Callable[[pd.Series], R | E | float]:
    """
    Returns the input function with additional information already supplied.

    The returned function takes an input with a subset of the indices present in the additional information (`args` and
    `kwargs`), fn is then called with additional information that matches the index of the input. If `ignore_nan` is set
    to `True`, indices where the input is `np.nan` are not passed to the fn. `empty_value` is returned if the index is
    empty of the input to x.
    """

    def f(x: pd.Series) -> R | E | float:
        idx = x.notna()[lambda x: x].index if ignore_nan else x.index
        if len(idx) == 0:
            return empty_value
        return fn(
            x.loc[idx],
            *(arg if arg is None else arg.loc[idx] for arg in args),
            **{k: v if v is None else v.loc[idx] for k, v in kwargs.items()},
        )

    return f
