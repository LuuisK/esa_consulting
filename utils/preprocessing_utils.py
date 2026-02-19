"""Module with utilities for preprocessing."""

from __future__ import annotations

import itertools
import warnings

import miceforest as mf
import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


def create_vars_info(var_labels: dict[str, str], value_labels: dict[str, dict[float, str]]) -> pd.DataFrame:
    """
    Creates a dataframe containing information about variables.
    """
    # Create data frame with variable labels
    vars_info = pd.DataFrame(var_labels.items(), columns=["name", "description"], dtype=pd.StringDtype())

    # Add dictionaries of value labels
    vars_info["labels"] = pd.NA
    for var in vars_info["name"]:
        if var in value_labels:
            vars_info.loc[vars_info["name"].eq(var), "labels"] = [value_labels[var]]
        else:
            vars_info.loc[vars_info["name"].eq(var), "labels"] = [{}]

    return vars_info


def sort_groups_vars_info(vars_info: pd.DataFrame, groups_by_name: dict[str, list[str]]) -> pd.DataFrame:
    """
    Partitions variables in vars_info into groups.
    """
    # Return a modified copy of the input
    vars_info = vars_info.copy()

    # Check that there are no multiply assigned variables
    all_grouped_vars = list(itertools.chain.from_iterable(groups_by_name.values()))
    duplicates = set([var for var in all_grouped_vars if all_grouped_vars.count(var) > 1])
    if len(duplicates) != 0:
        raise ValueError(f"Some variables have multiple group membership: {duplicates}.")

    # Display variables that are not assigned a group
    not_assigned = [var for var in vars_info["name"] if var not in all_grouped_vars]
    if len(not_assigned) != 0:
        print(f"Some variables are not assigned to a group: {duplicates}.")

    # Assign groups to variables
    vars_info["group"] = pd.Categorical(
        values=len(vars_info) * [pd.NA], categories=pd.Series(list(groups_by_name), dtype=pd.StringDtype())
    )
    for group_name, group_vars in groups_by_name.items():
        vars_info.loc[vars_info["name"].isin(group_vars), "group"] = group_name

    return vars_info


def sort_types_vars_info(vars_info: pd.DataFrame, types_by_name: dict[str, list[str]]) -> pd.DataFrame:
    """
    Sorts variables in vars_info into types.

    The type of a variable is one of: all_na, categorical, continuous, or NA in case of no type assignment.
    """
    # Return a modified copy of the input
    vars_info = vars_info.copy()

    # Check that there are no multiply assigned variables
    all_typed_vars = list(itertools.chain.from_iterable(types_by_name.values()))
    duplicates = set([var for var in all_typed_vars if all_typed_vars.count(var) > 1])
    if len(duplicates) != 0:
        raise ValueError(f"Some variables have several types: {duplicates}.")

    # Display variables that are not assigned a type
    not_assigned = [var for var in vars_info["name"] if var not in all_typed_vars]
    if len(not_assigned) != 0:
        print(f"Some variables are not assigned to a type: {duplicates}.")

    # Assign types to variables
    vars_info["type"] = pd.Categorical(
        values=len(vars_info) * [pd.NA], categories=pd.Series(list(types_by_name), dtype=pd.StringDtype())
    )
    for type_name, type_vars in types_by_name.items():
        vars_info.loc[vars_info["name"].isin(type_vars), "type"] = type_name

    return vars_info


def convert_data_types(data: pd.DataFrame, vars_info: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the data types if necessary.

    Converts factorial data to have categorical data type and saves the original levels in vars_info.
    """
    # Return a modified copy of the input
    data = data.copy()

    # Convert categorical variables to factors
    categorical_vars: pd.Series[pd.StringDtype] = vars_info.loc[
        vars_info["type"].isin(["binary", "nominal", "ordinal"]), "name"
    ]
    data[categorical_vars] = data[categorical_vars].astype("category")

    # # Recode binary to 0-1 in data and in vars_info["labels"]
    # binary_vars: pd.Series[pd.StringDtype] = vars_info.loc[vars_info["type"].eq("binary"), "name"]
    # not_0_1 = [var for var in binary_vars if set(data[var].cat.categories) != set((0, 1))]
    # for var in not_0_1:
    #     old_cats = data[var].cat.categories
    #     new_cats = [0, 1]
    #     old_value_dict = vars_info.loc[vars_info["name"].eq(var), "labels"].item()
    #     new_value_dict = {x: old_value_dict.get(old_cats[x], old_cats[x]) for x in new_cats}
    #     data[var] = data[var].cat.rename_categories(new_cats)
    #     vars_info.loc[vars_info["name"].eq(var), "labels"] = [new_value_dict]

    return data


def remove_variables(
    data: pd.DataFrame, vars_info: pd.DataFrame, drop_vars: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes specified features from both the data and variable information data-frames.
    """
    # Return a modified copy of the input
    data = data.copy()
    vars_info = vars_info.copy()

    if drop_vars is not None:
        # Remove specified unimportant variables
        data = data.drop(columns=drop_vars, errors="ignore")
        vars_info = vars_info[~vars_info["name"].isin(drop_vars)].reset_index()

    vars_info["type"] = vars_info["type"].cat.remove_unused_categories()
    vars_info["group"] = vars_info["group"].cat.remove_unused_categories()

    return data, vars_info


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
