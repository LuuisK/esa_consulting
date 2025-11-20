"""
Script for plotting the correlation / association.

Run scripts as interactive notebooks in the follwing order:
    preprocessing.py  ->  plot_corr_assoc.py  ->  dimensionality_reduction.py  ->  clustering.py
"""

import itertools
import typing
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotnine as p9
from scipy.stats import kruskal, spearmanr
from scipy.stats.contingency import association, chi2_contingency

from utils import check_save_path, load_data

pd.options.mode.copy_on_write = True


def correlation_ratio(*samples: pd.Series) -> float:
    """
    Computes eta squared (the squared correlation ratio).

    Outputs eta squared. Since this is not defined when all values in all samples are equal 1 is returned in that case.
    """
    # Calculate grand mean
    all_values = np.concatenate(samples)
    grand_mean = np.nanmean(all_values)

    # Calculate between and total sums of squares
    ss_between = np.sum([len(sample) * (np.nanmean(sample) - grand_mean) ** 2 for sample in samples], dtype=np.float64)
    ss_total = np.nansum((all_values - grand_mean) ** 2, dtype=np.float64)

    # Calculate eta-squared, set to 1 if all values are equal
    eta_squared = ss_between / ss_total if ss_total != 0 else 1.0
    correlation_ratio: np.float64 = np.sqrt(eta_squared)

    return correlation_ratio


def measure_corr_continuous(valuesi: pd.Series, valuesj: pd.Series, **kwargs) -> tuple[float, float]:
    """
    Measures the correlation of two continuous variables.

    Outputs the absolute Spearman correlation coefficient with its p-value. The correlation coefficient is not defined
    if either of the variables is constant (this includes the case of only one observation).
    """
    # Return NaN if there are no complete observations
    if (valuesi.isna() | valuesj.isna()).all():
        return np.nan, np.nan

    # Compute absolute Spearman correlation coefficient with p-value
    try:
        stat, p_value = spearmanr(valuesi, valuesj, nan_policy="omit")
    except ValueError:
        stat, p_value = np.nan, np.nan
    stat = typing.cast(np.float64, stat)
    p_value = typing.cast(np.float64, p_value)
    abs_corr: np.float64 = np.abs(stat)

    # Fix numerical errors that make p-value 1 instead of 0
    if stat > 1.0:
        p_value = np.float64(0.0)

    return abs_corr, p_value


def measure_assoc_mixed(
    valuesi: pd.Series,
    valuesj: pd.Series,
    which_continuous: Literal["i", "j"] | None,
) -> tuple[float, float]:
    """
    Measures the association of a nominal and a continuous variable.

    Outputs eta and the p-value of the Kruskal-Wallis test. The correlation Kruskal-Wallis test only works if there is
    more than one group and not all values are equal. Due to the assumption that the statistic has a chi square
    distribution, the number of samples in each group must not be too small. A typical rule is that each sample must
    have at least 5 measurements.
    """
    # Return NaN if there are no complete observations.
    if (valuesi.isna() | valuesj.isna()).all():
        return np.nan, np.nan

    # Distinguish between the variable types
    if which_continuous is not None:
        is_cati = which_continuous == "j"
    else:
        # Find out which is the categorical and which is the continuous variable
        is_cati = isinstance(valuesi.dtype, pd.CategoricalDtype)
    cat_values, cont_values = (valuesi, valuesj) if is_cati else (valuesj, valuesi)

    # Compute eta and conduct Kruskal-Wallis test
    groups = [cont_values[dum] for var, dum in pd.get_dummies(cat_values).items()]
    stat, p_value = kruskal(*groups, nan_policy="omit")
    p_value = typing.cast(np.float64, p_value)
    assoc = correlation_ratio(*groups)

    return assoc, p_value


def measure_assoc_nominal(valuesi: pd.Series, valuesj: pd.Series, **kwargs) -> tuple[float, float]:
    """
    Measures the association of two nominal variables.

    Outputs Cramér's V and the corresponding p-value (from the chi-squared test). Doesn't work if either of the
    variables has only one category or a category without observations. An often quoted guideline for the validity of
    this calculation is that the test should be used only if the observed and expected frequencies in each cell are at
    least 5.
    """
    # Return NaN if there are no complete observations.
    if (valuesi.isna() | valuesj.isna()).all():
        return np.nan, np.nan

    # Compute Cramér's V and conduct chi-squared test
    cont_table = pd.crosstab(valuesi, valuesj)
    stat, p_value, dof, expected_freq = chi2_contingency(cont_table)
    stat = typing.cast(np.float64, stat)
    p_value = typing.cast(np.float64, p_value)
    assoc = association(observed=cont_table, method="cramer")

    return assoc, p_value


def create_corr_assoc_matrix(
    data: pd.DataFrame,
    treat_cat_as_continuous: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a matrix containing values that measure the correlation or association between variables.

    The correlation / association matrix includes (all ranging from 0 to 1):
    - the absolute value of Spearman's rank correlation coefficient for quantitative-quantitative variable pairs,
    - the correlation ratio for quantitative-qualitative variable pairs,
    - Cramér's V for qualitative-qualitative variable pairs.

    The p-value matrix contains test results from:
    - Spearman's rank correlation test for quantitative-quantitative variable pairs,
    - the Kruskal-Wallis test for quantitative-qualitative variable pairs,
    - the chi squared test for qualitative-qualitative variable pairs.
    """
    # Copy to not modify the input
    data = data.copy()

    # Initialize matrices
    corr_assoc_matrix = pd.DataFrame(np.eye(len(data.columns)), columns=data.columns, index=data.columns)
    p_value_matrix = pd.DataFrame(np.eye(len(data.columns)), columns=data.columns, index=data.columns)

    # Recode to continuous (category codes go up from 0, -1 means pd.NA)
    treat_cat_as_continuous = [] if treat_cat_as_continuous is None else treat_cat_as_continuous
    treat_cat_as_continuous = [var for var in treat_cat_as_continuous if var in data.columns]
    data[treat_cat_as_continuous] = data[treat_cat_as_continuous].apply(lambda col: col.cat.codes.replace(-1, np.nan))

    # Loop over pairs of variables
    for vari, varj in itertools.combinations(data.columns, r=2):
        # Get variable types
        which_continuous = ""
        if not isinstance(data[vari].dtype, pd.CategoricalDtype):
            which_continuous += "i"
        if not isinstance(data[varj].dtype, pd.CategoricalDtype):
            which_continuous += "j"
        n_cont = len(which_continuous)

        # Calculate different association / correlation measures depending on variable types
        corr_assoc_func = [measure_assoc_nominal, measure_assoc_mixed, measure_corr_continuous][n_cont]
        corr_assoc, p_value = corr_assoc_func(data[vari], data[varj], which_continuous=which_continuous)

        # Fill in matrices
        corr_assoc_matrix.loc[vari, varj] = corr_assoc_matrix.loc[varj, vari] = corr_assoc
        p_value_matrix.loc[vari, varj] = p_value_matrix.loc[varj, vari] = p_value

    return corr_assoc_matrix, p_value_matrix


def _is_cat(var: str, values: pd.Series, treat_cat_as_continuous: list[str] | None = None):
    return isinstance(values.dtype, pd.CategoricalDtype) and not (
        treat_cat_as_continuous is not None and var in treat_cat_as_continuous
    )


def plot_corr_assoc_matrix(data: pd.DataFrame, treat_cat_as_continuous: list[str] | None = None) -> p9.ggplot:
    """
    Creates and displays a matrix containing values that measure the correlation or association between variables.

    See create_corr_assoc_matrix for the correlation and association measures used.
    """
    # Create correlation matrix
    corr_assoc_matrix, p_value_matrix = create_corr_assoc_matrix(data, treat_cat_as_continuous=treat_cat_as_continuous)

    # Order variables into blocks
    categorical_vars = [var for var in data.columns if _is_cat(var, data[var], treat_cat_as_continuous)]
    continuous_vars = set(data.columns) - set(categorical_vars)
    vars_ordered = [*continuous_vars, *categorical_vars]

    # Pivot matrices
    corr_assocs = corr_assoc_matrix.stack(future_stack=True).rename("corr_assoc")  # type: ignore
    p_values = p_value_matrix.stack(future_stack=True).rename("p_value")  # type: ignore

    # Create dataset
    def _get_type(var, values, treat_cat_as_continuous: list[str] | None = None):
        return "qualitative" if _is_cat(var, values, treat_cat_as_continuous) else "quantitative"

    data_ggplot = (
        pd.merge(corr_assocs, p_values, left_index=True, right_index=True)
        .reset_index(names=["vari", "varj"])
        .astype(
            {
                "vari": pd.CategoricalDtype(categories=vars_ordered),
                "varj": pd.CategoricalDtype(categories=vars_ordered[::-1]),
                "corr_assoc": np.float64,
                "p_value": np.float64,
            }
        )
        .assign(
            **{
                "i_type": lambda x: pd.Categorical(
                    x.apply(lambda row: _get_type(row["vari"], data[row["vari"]], treat_cat_as_continuous), axis=1),
                    categories=["quantitative", "qualitative"],
                ),
                "j_type": lambda x: pd.Categorical(
                    x.apply(lambda row: _get_type(row["varj"], data[row["varj"]], treat_cat_as_continuous), axis=1),
                    categories=["quantitative", "qualitative"],
                ),
                "significant": lambda x: pd.Categorical((x["p_value"] < 0.05).astype(str), categories=["False"]),
            }
        )
    )

    # Fix plot labels
    var_scale = data_ggplot["i_type"].unique()
    if "qualitative" not in var_scale:
        corr_assoc_str = "correlation"
        n_types = 1
    elif "quantitative" not in var_scale:
        corr_assoc_str = "association"
        n_types = 1
    else:
        corr_assoc_str = "correlation /\nassociation"
        n_types = 2

    # Create plot
    corr_assoc_plot = (
        p9.ggplot(data_ggplot, p9.aes(x="vari", y="varj", fill="corr_assoc"))
        + p9.geom_tile(width=1)
        + (p9.facet_grid(rows="j_type", cols="i_type", scales="free", space="free") if n_types == 2 else None)
        + p9.labs(x="", y="", fill=corr_assoc_str)
        + p9.scale_fill_gradient(low="darkblue", high="lightblue", na_value="red", limits=(0, 1))
        + p9.coord_fixed()
        + p9.geom_point(p9.aes(x="vari", y="varj", shape="significant"), inherit_aes=False)
        + p9.scale_shape_manual(
            values={"False": "x"},
            name=" ",
            labels={"False": f"no significant\n{corr_assoc_str}"},
            na_translate=False,
        )
        + p9.theme_538()
        + p9.theme(axis_text_x=p9.element_text(rotation="vertical"))
    )

    return corr_assoc_plot


if __name__ == "__main__":
    # Hardcoded paths, change to apply to new data
    data_path = Path("data/preprocessed/esa_2018_2021.pkl")
    vars_info_path = Path("data/preprocessed/esa_vars_info.pkl")

    image_folder = "outputs/corr_assoc_plots"
    plot_save_name_template = "corr_assoc_plot_{group}.png"

    allow_overwrite = True

    # Read in data
    data: pd.DataFrame = load_data(data_path)
    vars_info: pd.DataFrame = load_data(vars_info_path)

    # Get variable groups
    var_groups = {
        group: vars_info.loc[vars_info["group"] == group, "name"] for group in vars_info["group"].cat.categories
    }
    var_groups = typing.cast(dict[str, list[str]], var_groups)

    # Visualize correlation matrices
    treat_cat_as_continuous: list[str] = vars_info.loc[vars_info["type"].isin(["binary", "ordinal"]), "name"].tolist()

    for group, vars in var_groups.items():
        print(f"{group=}, {[var for var in vars]}")
        plot = plot_corr_assoc_matrix(data[vars], treat_cat_as_continuous)
        plot.show()
        plot_save_path = Path(image_folder, plot_save_name_template.format(group=group))
        check_save_path(plot_save_path, allow_overwrite=allow_overwrite)
        plot.save(plot_save_path)
        print(f"Saved {group} correlation / association plot as PNG to {plot_save_path}.")
