"""
Script for plotting the correlation / association.

Run scripts as interactive notebooks in the follwing order:
    preprocessing.py  ->  plot_corr_assoc.py  ->  dimensionality_reduction.py  ->  clustering.py
"""

import os

import numpy as np
import pandas as pd
import plotnine as p9
from scipy.stats import kruskal, spearmanr
from scipy.stats.contingency import association, chi2_contingency

from utils import check_save_path

pd.options.mode.copy_on_write = True


# Hardcoded paths, change to apply to new data
data_folder = os.path.join("data", "preprocessed")
data_path = os.path.join(data_folder, "esa_2018_2021.pkl")
vars_info_path = os.path.join(data_folder, "esa_vars_info.pkl")

image_folder = os.path.join("outputs", "images")
plot_save_path_template = os.path.join(image_folder, "corr_assoc_plot_{}.png")

allow_overwrite = True


def eta_squared(*samples: pd.Series) -> float:
    """
    Computes eta squared (the squared correlation ratio).

    Outputs eta squared. Since this is not defined when all values in all samples are equal 1 is returned in that case.
    """
    # Calculate grand mean
    all_values = np.concatenate(samples)
    grand_mean = np.nanmean(all_values)

    # Calculate between and total sums of squares
    ss_between = np.sum([len(sample) * (np.nanmean(sample) - grand_mean) ** 2 for sample in samples])
    ss_total = np.nansum((all_values - grand_mean) ** 2)

    # Calculate eta-squared, set to 1 if all values are equal
    eta_squared = ss_between / ss_total if ss_total != 0 else 1.0

    return eta_squared


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
    stat, p_value = spearmanr(valuesi, valuesj, nan_policy="omit")
    abs_corr = np.abs(stat)

    # Fix numerical errors that make p-value 1 instead of 0
    if stat > 1.0:
        p_value = 0.0

    return abs_corr, p_value


def measure_assoc_mixed(valuesi: pd.Series, valuesj: pd.Series) -> tuple[float, float]:
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

    # Find out which is the categorical and which is the continuous variable
    is_cati = isinstance(valuesi.dtype, pd.CategoricalDtype)
    cat_values, cont_values = (valuesi, valuesj) if is_cati else (valuesj, valuesi)

    # Compute eta and conduct Kruskal-Wallis test
    groups = [cont_values[dum] for var, dum in pd.get_dummies(cat_values).items()]
    stat, p_value = kruskal(*groups, nan_policy="omit")
    assoc = np.sqrt(eta_squared(*groups))
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
    # Initialize matrices
    corr_assoc_matrix = pd.DataFrame(columns=data.columns, index=data.columns)
    p_value_matrix = pd.DataFrame(columns=data.columns, index=data.columns)

    # Loop over pairs of variables
    for vari, valuesi in data.items():
        for varj, valuesj in data.items():
            # Get variable types
            is_cati = isinstance(valuesi.dtype, pd.CategoricalDtype)
            is_catj = isinstance(valuesj.dtype, pd.CategoricalDtype)
            n_cat = is_cati + is_catj

            # Calculate different association / correlation measures depending on variable types
            corr_assoc_func = [measure_corr_continuous, measure_assoc_mixed, measure_assoc_nominal][n_cat]
            corr_assoc, p_value = corr_assoc_func(valuesi, valuesj)

            # Fill in matrices
            corr_assoc_matrix.loc[vari, varj] = corr_assoc
            p_value_matrix.loc[vari, varj] = p_value

    # Return matrices
    return corr_assoc_matrix, p_value_matrix


def _get_type(values):
    return "qualitative" if isinstance(values.dtype, pd.CategoricalDtype) else "quantitative"


def plot_corr_assoc_matrix(data: pd.DataFrame) -> p9.ggplot:
    # Create correlation matrix
    corr_assoc_matrix, p_value_matrix = create_corr_assoc_matrix(data)

    # Order variables into blocks
    categorical_vars = data.columns[[isinstance(dtype, pd.CategoricalDtype) for dtype in data.dtypes]]
    continuous_vars = set(data.columns) - set(categorical_vars)
    vars_ordered = [*continuous_vars, *categorical_vars]

    # Pivot matrices
    corr_assocs = corr_assoc_matrix.stack(future_stack=True).rename("corr_assoc")
    p_values = p_value_matrix.stack(future_stack=True).rename("p_value")

    # Create dataset
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
                    x.apply(lambda row: _get_type(data[row["vari"]]), axis=1),
                    categories=["quantitative", "qualitative"],
                ),
                "j_type": lambda x: pd.Categorical(
                    x.apply(lambda row: _get_type(data[row["varj"]]), axis=1),
                    categories=["quantitative", "qualitative"],
                ),
            }
        )
    )

    # Create plot
    corr_assoc_plot = (
        p9.ggplot(data_ggplot, p9.aes(x="vari", y="varj", fill="corr_assoc"))
        + p9.geom_tile()
        + p9.facet_grid(rows="j_type", cols="i_type", scales="free", space="free")
        + p9.labs(x="", y="", fill="correlation /\nassociation")
        + p9.scale_fill_gradient(low="darkblue", high="lightblue", na_value="red", limits=[0, 1])
        + p9.theme_538()
        + p9.theme(axis_text_x=p9.element_text(rotation="vertical"))
        + p9.coord_fixed()
        + p9.geom_point(
            p9.aes(x="vari", y="varj", shape=pd.Categorical([0], categories=[0])),
            data=data_ggplot[~(data_ggplot["p_value"] < 0.05)],
        )
        + p9.scale_shape_manual(
            values={0: "x"},
            labels={0: "no significant\ncorrelation /\nassociation"},
        )
    )
    return corr_assoc_plot


if __name__ == "__main__":
    # Read in data
    data: pd.DataFrame = pd.read_pickle(data_path)
    vars_info: pd.DataFrame = pd.read_pickle(vars_info_path)

    # Get variable groups
    var_groups: dict[str, list[str]] = {
        group: vars_info.loc[vars_info["group"] == group, "variable"] for group in vars_info["group"].cat.categories
    }

    # Visualize correlation matrices
    var_groups_iter = iter(var_groups.items())
    # To show and save group correlations repeat code below
    group, vars = next(var_groups_iter)
    print(f"{group=}, {vars=}")
    plot = plot_corr_assoc_matrix(data[vars])
    plot.show()
    plot_save_path = plot_save_path_template.format(group)
    check_save_path(plot_save_path, allow_overwrite=allow_overwrite)
    plot.save(plot_save_path)
    print(f"Saved correlation / association plot as PNG to {plot_save_path}.")
