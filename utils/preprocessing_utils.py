"""Module with utilities for preprocessing."""

from __future__ import annotations

import ast
import builtins
import itertools
from typing import Any

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

    # Add string repr of dictionaries of value labels, so it can be serialized as parquet
    data.meta["labels"] = pd.NA
    for var in data.meta["name"]:
        if var in value_labels:
            data.meta.loc[lambda x: x["name"].eq(var), "labels"] = [
                repr({k.item() if isinstance(k, np.number) else float(k): v for k, v in value_labels[var].items()})
            ]
        else:
            data.meta.loc[lambda x: x["name"].eq(var), "labels"] = [repr({})]

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
    duplicates = {var for var in all_grouped_vars if all_grouped_vars.count(var) > 1}
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
    duplicates = {var for var in all_typed_vars if all_typed_vars.count(var) > 1}
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


def impute_allowed_skip(data: DatasetBundle, imp_val: dict[str, Any]) -> DatasetBundle:
    """Impute allowed skips specified in imp_val with their default value."""
    # Return a modified copy of the input
    data = data.copy()

    if data.na is None:
        raise ValueError("'data.na' may not be None.")

    imp_count = 0
    for var, val in imp_val.items():
        mask = data.na[var].eq("allowed skip")

        data.data.loc[mask, var] = val

        imputed_cat = "imputed skip"
        if imputed_cat not in data.na[var].cat.categories:
            data.na[var] = data.na[var].cat.add_categories([imputed_cat])
        data.na.loc[mask, var] = imputed_cat

        imp_count += mask.sum()

    print(f"Imputed {imp_count} allowed skips.")

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
        labels_old = ast.literal_eval(data.meta.loc[lambda x: x["name"].eq(var), "labels"].item())
        labels_new = {cat_map.get(key, key): value for key, value in labels_old.items()}
        data.meta.loc[lambda x: x["name"].eq(var), "labels"] = [repr(labels_new)]

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
        data.meta = data.meta.loc[lambda x: ~x["name"].isin(drop_vars)].reset_index(drop=True)
        data.meta["type"] = data.meta["type"].cat.remove_unused_categories()
        data.meta["group"] = data.meta["group"].cat.remove_unused_categories()

    if data.na is not None:
        data.na = data.na.drop(columns=drop_vars, errors="ignore")

    return data


def pool_ordinal_categories_and_remove_unsupported_variables(
    data: DatasetBundle, *, min_count: int = 30, min_weighted_count: float = 20, weight_var: str = "wichtn"
) -> tuple[DatasetBundle, pd.DataFrame]:
    """
    Pool sparse ordinal categories and remove unsupported categorical variables.

    The lowest ordinal category is assumed to be the baseline and will not be pooled with other categories. Pooling
    iteratively pools categories without sufficient support into lower neighboring ones. Subsequently all categorical
    variables are checked for their support and removed if they don't meet the specified thresholds in all their
    categories.
    """
    data = data.copy()
    if data.meta is None:
        raise ValueError("'data.meta' may not be None.")

    if min_count < 1 or not np.isfinite(min_weighted_count) or min_weighted_count < 0:
        raise ValueError("min_count must be positive and min_weighted_count must be finite and non-negative.")

    # Normalize weights to make weighted support comparable to raw counts
    sample_weight = data.data[weight_var]
    sample_weight = sample_weight * len(sample_weight) / sample_weight.sum()

    categorical_vars = data.data.select_dtypes("category").columns.to_list()
    ordinal_vars: list[str] = data.meta.loc[
        lambda x: x["type"].eq("ordinal") & x["name"].isin(data.data.columns), "name"
    ].to_list()
    category_maps: dict[str, dict[Any, Any]] = {
        var: {category: category for category in data.data[var].cat.categories} for var in categorical_vars
    }

    # Pool adjacent ordinal categories while keeping the first category separate
    for var in ordinal_vars:
        values = data.data[var]
        if not isinstance(values.dtype, pd.CategoricalDtype) or not values.cat.ordered:
            raise ValueError(f"'{var}' must be an ordered categorical variable.")

        categories = values.cat.categories.to_list()
        if len(categories) == 0:
            continue

        counts = {category: values.eq(category).sum() for category in categories}
        weighted_counts = {category: sample_weight.loc[values.eq(category)].sum() for category in categories}

        def has_support(bin_categories: list[Any]) -> bool:
            return (
                sum(counts[category] for category in bin_categories) >= min_count
                and sum(weighted_counts[category] for category in bin_categories) >= min_weighted_count
            )

        category_bins = [[categories[0]]]
        next_category: list[Any] = []
        for category in categories[1:]:
            next_category.append(category)
            if has_support(next_category):
                category_bins.append(next_category)
                next_category = []

        # Merge an unsupported upper tail into the preceding non-protected bin
        if len(next_category) != 0:
            if len(category_bins) == 1:
                category_bins.append(next_category)
            else:
                category_bins[-1].extend(next_category)

        category_bins_by_name = {f"{bin[0]}-{bin[-1]}" if len(bin) > 1 else bin[0]: bin for bin in category_bins}
        pooling_map = {category: name for name, bin in category_bins_by_name.items() for category in bin}
        labels = ast.literal_eval(data.meta.loc[lambda x: x["name"].eq(var), "labels"].item())
        if not isinstance(labels, dict):
            raise ValueError(f"Labels for '{var}' must be a dictionary.")

        new_categories = list(category_bins_by_name)
        new_labels = {
            name: " / ".join(str(labels.get(category, category)) for category in category_bin)
            for name, category_bin in category_bins_by_name.items()
        }
        data.data[var] = values.map(pooling_map, na_action="ignore").astype(
            pd.CategoricalDtype(categories=new_categories, ordered=True)
        )
        data.meta.loc[lambda x: x["name"].eq(var), "labels"] = repr(new_labels)
        category_maps[var] = pooling_map

    # Remove categorical variables with fewer than two supported observed categories
    unsupported_vars: list[str] = []
    for var in categorical_vars:
        values = data.data[var]
        counts = values.value_counts(sort=False)
        weighted_counts = sample_weight.groupby(values, observed=False).sum()
        observed_categories = [category for category in values.cat.categories if counts[category] > 0]
        is_supported = len(observed_categories) >= 2 and all(
            counts[category] >= min_count and weighted_counts[category] >= min_weighted_count
            for category in observed_categories
        )
        if not is_supported:
            unsupported_vars.append(var)

    unsupported_vars_set = set(unsupported_vars)
    category_mapping_rows = [
        {
            "variable": var,
            "original_category": original_category,
            "pooled_category": pd.NA if var in unsupported_vars_set else pooled_category,
        }
        for var, category_map in category_maps.items()
        for original_category, pooled_category in category_map.items()
    ]
    category_mapping = pd.DataFrame(category_mapping_rows, columns=["variable", "original_category", "pooled_category"])
    return remove_variables(data, unsupported_vars), category_mapping
