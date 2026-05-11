"""Module with utilities for analysing correlations and associations."""

from __future__ import annotations

import itertools
import typing
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from scipy.stats.contingency import association, chi2_contingency

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


def correlation_ratio(*samples: pd.Series) -> float:
    """
    Computes eta (the correlation ratio).

    Outputs eta. Since this is not defined when all values in all samples are equal 1 is returned in that case.
    """
    # Calculate grand mean
    all_values = np.concatenate(samples)
    grand_mean = np.nanmean(all_values)

    # Calculate between and total sums of squares
    ss_between = np.sum([sample.count() * (np.nanmean(sample) - grand_mean) ** 2 for sample in samples], dtype="float")
    ss_total = np.nansum((all_values - grand_mean) ** 2, dtype="float")

    # Calculate eta-squared, set to 1 if all values are equal
    eta_squared = ss_between / ss_total if ss_total != 0 else 1.0
    correlation_ratio: np.float64 = np.sqrt(eta_squared)

    return correlation_ratio


def measure_corr_continuous(valuesi: pd.Series, valuesj: pd.Series) -> tuple[float, float]:
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
    valuesi: pd.Series, valuesj: pd.Series, which_continuous: Literal["i", "j"] | None = None
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
    obs = np.concat(groups)
    if len(obs) == 0 or np.all(obs == next(iter(obs))):
        # Kruskal-Wallis test is not defined if all values are equal. We define the p-value as 1.
        p_value = 1.0
    else:
        stat, p_value = kruskal(*groups, nan_policy="omit")
    p_value = typing.cast(np.float64, p_value)
    assoc = correlation_ratio(*groups)

    return assoc, p_value


def measure_assoc_nominal(valuesi: pd.Series, valuesj: pd.Series) -> tuple[float, float]:
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


def create_corr_assoc_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    # Loop over pairs of variables
    for vari, varj in itertools.combinations(data.columns, r=2):
        # Calculate different association / correlation measures depending on variable types
        is_cat = (isinstance(data[vari].dtype, pd.CategoricalDtype), isinstance(data[varj].dtype, pd.CategoricalDtype))
        if all(is_cat):
            corr_assoc_func = measure_assoc_nominal
        elif any(is_cat):
            corr_assoc_func = partial(measure_assoc_mixed, which_continuous="j" if is_cat[0] else "i")
        else:
            corr_assoc_func = measure_corr_continuous
        corr_assoc, p_value = corr_assoc_func(data[vari], data[varj])

        # Fill in matrices
        corr_assoc_matrix.loc[vari, varj] = corr_assoc_matrix.loc[varj, vari] = corr_assoc
        p_value_matrix.loc[vari, varj] = p_value_matrix.loc[varj, vari] = p_value

    return corr_assoc_matrix, p_value_matrix
