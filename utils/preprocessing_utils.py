"""Module with utilities for preprocessing."""

from __future__ import annotations

import builtins
import itertools
import warnings
from collections.abc import Iterable
from typing import Any

import miceforest as mf
import numpy as np
import pandas as pd

from utils.saving_utils import DatasetBundle

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


def create_meta(
    data: DatasetBundle, var_labels: dict[str, str], value_labels: dict[str, dict[float, str]]
) -> DatasetBundle:
    """
    Creates a dataframe containing information about variables.
    """
    # Create data frame with variable labels
    data.meta = pd.DataFrame(var_labels.items(), columns=pd.Index(["name", "description"]), dtype="str")

    # Add dictionaries of value labels
    data.meta["labels"] = pd.NA
    for var in data.meta["name"]:
        if var in value_labels:
            data.meta.loc[lambda x: x["name"].eq(var), "labels"] = [value_labels[var]]
        else:
            data.meta.loc[lambda x: x["name"].eq(var), "labels"] = [{}]

    return data


def sort_groups_meta(data: DatasetBundle, groups_by_name: dict[str, list[str]]) -> DatasetBundle:
    """
    Partitions variables into groups.
    """
    # Return a modified copy of the input
    data = data.copy()

    if data.meta is None:
        raise ValueError("'data.meta' may not be None.")

    # Check that there are no multiply assigned variables
    all_grouped_vars = list(itertools.chain.from_iterable(groups_by_name.values()))
    duplicates = set([var for var in all_grouped_vars if all_grouped_vars.count(var) > 1])
    if len(duplicates) != 0:
        raise ValueError(f"Some variables have multiple group membership: {duplicates}.")

    # Display variables that are not assigned a group
    not_assigned = [var for var in data.meta["name"] if var not in all_grouped_vars]
    if len(not_assigned) != 0:
        print(f"Some variables are not assigned to a group: {duplicates}.")

    # Assign groups to variables
    data.meta["group"] = pd.Categorical(
        values=len(data.meta) * [pd.NA], categories=pd.Series(list(groups_by_name), dtype="str")
    )
    for group_name, group_vars in groups_by_name.items():
        data.meta.loc[lambda x: x["name"].isin(group_vars), "group"] = group_name

    return data


def sort_types_meta(data: DatasetBundle, types_by_name: dict[str, list[str]]) -> DatasetBundle:
    """
    Sorts variables into types.

    The type of a variable is one of: all_na, categorical, continuous, or NA in case of no type assignment.
    """
    # Return a modified copy of the input
    data = data.copy()

    if data.meta is None:
        raise ValueError("'data.meta' may not be None.")

    # Check that there are no multiply assigned variables
    all_typed_vars = list(itertools.chain.from_iterable(types_by_name.values()))
    duplicates = set([var for var in all_typed_vars if all_typed_vars.count(var) > 1])
    if len(duplicates) != 0:
        raise ValueError(f"Some variables have several types: {duplicates}.")

    # Display variables that are not assigned a type
    not_assigned = [var for var in data.meta["name"] if var not in all_typed_vars]
    if len(not_assigned) != 0:
        print(f"Some variables are not assigned to a type: {duplicates}.")

    # Assign types to variables
    data.meta["type"] = pd.Categorical(
        values=len(data.meta) * [pd.NA], categories=pd.Series(list(types_by_name), dtype="str")
    )
    for type_name, type_vars in types_by_name.items():
        data.meta.loc[lambda x: x["name"].isin(type_vars), "type"] = type_name

    return data


def create_missing_info(data: DatasetBundle, na_labels: dict[Any, str]) -> DatasetBundle:
    """
    Creates a dataframe containing missingness information.

    Encodes missingness with NA and saves missingness metadata in another dataframe.
    """
    # Create data frame with missingness labels
    data.na = data.data.apply(
        lambda x: (
            x.map(na_labels)
            .fillna("observed")
            .astype(pd.CategoricalDtype(categories=pd.Index(["observed", *na_labels.values()], dtype="str")))
        )
    )

    # Replace missing values by NA
    data.data = data.data.replace(list(na_labels), pd.NA)
    return data


# def impute_allowed_skip(data: DatasetBundle, imp_val: dict[str, Any] | None = None) -> DatasetBundle:
#     """
#     Impute allowed skips of a dataset with a default value.

#     By default imputs the value 0. Other values need to be specified in `imp_val`.
#     """
#     # Return a modified copy of the input
#     data = data.copy()

#     if data.na is None:
#         raise ValueError("'data.na' may not be None.")

#     imp_val = {} if imp_val is None else imp_val

#     for var in data.data.columns:
#         mask = data.na[var].eq("allowed skip")

#         data.data.loc[mask, var] = imp_val.get(var, 0)

#         imputed_cat = "imputed skip"
#         if imputed_cat not in data.na[var].cat.categories:
#             data.na[var] = data.na[var].cat.add_categories([imputed_cat])
#         data.na.loc[mask, var] = imputed_cat

#     return data

def impute_allowed_skip(data: DatasetBundle, imp_val: dict[str, Any]) -> DatasetBundle:
    """Impute allowed skips specified in imp_val with their default value.
    """
    # Return a modified copy of the input
    data = data.copy()

    if data.na is None:
        raise ValueError("'data.na' may not be None.")

    for var, val in imp_val.items():
        mask = data.na[var].eq("allowed skip")

        data.data.loc[mask, var] = val

        imputed_cat = "imputed skip"
        if imputed_cat not in data.na[var].cat.categories:
            data.na[var] = data.na[var].cat.add_categories([imputed_cat])
        data.na.loc[mask, var] = imputed_cat

    return data


def convert_data_types(data: DatasetBundle) -> DatasetBundle:
    """
    Converts the data types if necessary.

    Converts factorial data to have categorical data type and saves the original levels in vars_info. (instead of
    letting StataReader do this, hence convert_categoricals=False in pd.read_stata)
    """
    # Return a modified copy of the input
    data = data.copy()

    if data.meta is None:
        raise ValueError("'data.meta' may not be None.")

    # Convert categorical variables to factors with string categories
    categorical_vars: list[str] = data.meta.loc[
        lambda x: x["type"].isin(["binary", "nominal", "ordinal"]), "name"
    ].to_list()
    ordered_vars: list[str] = data.meta.loc[lambda x: ~x["type"].eq("nominal"), "name"].to_list()
    # ordered_vars: list[str] = data.meta.loc[lambda x: x["type"].eq("ordinal"), "name"].to_list()  # TODO: binary too?
    for var in categorical_vars:
        # Change categories to strings
        cats_old = data.data[var].drop_duplicates().dropna().sort_values()
        cats_new = pd.Index(cats_old, dtype="str")
        cat_dtype = pd.CategoricalDtype(categories=cats_new, ordered=var in ordered_vars)
        cat_map = dict(zip(cats_old, cats_new, strict=True))

        # Convert data type
        data.data[var] = data.data[var].map(cat_map).astype(cat_dtype)

        # Update keys in label dictionary
        labels_old = data.meta.loc[lambda x: x["name"].eq(var), "labels"].item()
        labels_new = {cat_map.get(key, key): value for key, value in labels_old.items()}
        data.meta.loc[lambda x: x["name"].eq(var), "labels"] = [labels_new]

    return data


def get_description(meta: pd.DataFrame, var: str, slice: int | slice | None = None) -> str:
    """Get the sliced description of a variable."""
    if slice is None:
        slice = builtins.slice(None, None, None)
    elif isinstance(slice, int):
        slice = builtins.slice(slice, slice + 1)
    description = meta.loc[lambda y: y["name"].eq(var), "description"].item()
    description_sliced = description.split()[slice]
    return " ".join(description_sliced)


def remove_variables(data: DatasetBundle, drop_vars: list[str] | None = None) -> DatasetBundle:
    """
    Removes specified features from both the data.
    """
    # Return a modified copy of the input
    data = data.copy()

    if drop_vars is None:
        return data

    # Remove specified unimportant variables
    data.data = data.data.drop(columns=drop_vars, errors="ignore")

    if data.meta is not None:
        data.meta = data.meta[lambda x: ~x["name"].isin(drop_vars)].reset_index(drop=True)
        data.meta["type"] = data.meta["type"].cat.remove_unused_categories()
        data.meta["group"] = data.meta["group"].cat.remove_unused_categories()

    if data.na is not None:
        data.na = data.na.drop(columns=drop_vars, errors="ignore")

    return data


def impute_data(
    data: pd.DataFrame,
    rng: np.random.Generator | np.random.BitGenerator | None = None,
    mi_number: int = 1,
    mice_iterations: int = 3,
    # ordinal_variables = ..., # TODO: miceforest deals with ordinal variables as nominal
    **kwargs,
) -> mf.ImputationKernel:
    """
    Imputes data using MICE.
    """
    # Create RNG to simulate randomness
    rng = np.random.default_rng(rng)

    # Filter keyword arguments
    kernel_kwargs = {
        key: value for key, value in kwargs.items() if key in mf.ImputationKernel.__init__.__code__.co_varnames
    }
    mice_kwargs = {key: value for key, value in kwargs.items() if key not in kernel_kwargs}

    # Change PMM strategy for categorical variables to avoid numerical errors. See
    # https://github.com/AnotherSamWilson/miceforest/issues/96#issuecomment-3242644783.
    categorical_vars = data.select_dtypes("category").columns
    pmm_strategies = {var: "shap" if var in categorical_vars else "normal" for var in data.columns}

    # Impute
    imputation_kernel = mf.ImputationKernel(
        data,
        num_datasets=mi_number,
        random_state=np.random.RandomState(rng.bit_generator),
        mean_match_strategy=pmm_strategies,
        mean_match_candidates=5,
        **kernel_kwargs,
    )
    # Catch PerformanceWarning from DataFrame fracturing. Might still lead to crash if mice_iterations is set higher.
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        imputation_kernel.mice(iterations=mice_iterations, **mice_kwargs)

    return imputation_kernel
