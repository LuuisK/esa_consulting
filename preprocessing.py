"""
Script for preprocessing (including imputation).

Run scripts as interactive notebooks in the follwing order:
    preprocessing.py  ->  plot_corr_assoc.py  ->  dimensionality_reduction.py  ->  clustering.py
"""

import itertools
import typing
import warnings
from pathlib import Path

import miceforest as mf
import numpy as np
import pandas as pd

from utils import save_data

pd.options.mode.copy_on_write = True


def create_vars_info(var_labels: dict[str, str], value_labels: dict[str, dict[float, str]]) -> pd.DataFrame:
    """
    Creates a dataframe containing information about variables.
    """
    # Create data frame with variable labels
    vars_info = pd.DataFrame(var_labels.items(), columns=["name", "description"], dtype=pd.StringDtype())

    # Add dictionaries of value labels
    vars_info["labels"] = pd.NA
    for var in vars_info["name"]:
        if var in value_labels.keys():
            vars_info.loc[vars_info["name"] == var, "labels"] = [value_labels[var]]
        else:
            vars_info.loc[vars_info["name"] == var, "labels"] = [{}]

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
        values=len(vars_info) * [pd.NA], categories=pd.Series(groups_by_name.keys(), dtype=pd.StringDtype())
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
        values=len(vars_info) * [pd.NA], categories=pd.Series(types_by_name.keys(), dtype=pd.StringDtype())
    )
    for type_name, type_vars in types_by_name.items():
        vars_info.loc[vars_info["name"].isin(type_vars), "type"] = type_name

    return vars_info


def preprocess_data(
    data: pd.DataFrame, vars_info: pd.DataFrame, remove_all_na_vars: bool = True, drop_vars: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the data:
    - Converts factorial data to have categorical data type and saves the original levels in vars_info.
    - Removes unimportant features.
    """
    # Return a modified copy of the input
    data = data.copy()
    vars_info = vars_info.copy()

    # Convert categorical variables to factors
    categorical_vars: pd.Series[pd.StringDtype] = vars_info.loc[
        vars_info["type"].isin(["binary", "nominal", "ordinal"]), "name"
    ]
    data[categorical_vars] = data[categorical_vars].astype("category")

    # # Recode binary to 0-1 in data and in vars_info["labels"]
    # binary_vars: pd.Series[pd.StringDtype] = vars_info.loc[vars_info["type"] == "binary", "name"]
    # not_0_1 = [var for var in binary_vars if set(data[var].cat.categories) != set((0, 1))]
    # for var in not_0_1:
    #     old_cats = data[var].cat.categories
    #     new_cats = [0, 1]
    #     old_value_dict = vars_info.loc[vars_info["name"] == var, "labels"].item()
    #     new_value_dict = {x: old_value_dict.get(old_cats[x], old_cats[x]) for x in new_cats}
    #     data[var] = data[var].cat.rename_categories(new_cats)
    #     vars_info.loc[vars_info["name"] == var, "labels"] = [new_value_dict]

    if remove_all_na_vars:
        # Remove features that are NA everywhere from the dataset and vars_info
        all_na_vars: pd.Series[pd.StringDtype] = vars_info.loc[vars_info["type"] == "all_na", "name"]
        print(
            f"Dropped the following variables from the dataset because they are missing everywhere: "
            f"{all_na_vars.to_list()}"
        )
        data = data.drop(columns=all_na_vars)
        vars_info = vars_info[vars_info["type"] != "all_na"]
        vars_info["type"] = vars_info["type"].cat.remove_categories("all_na")

    if drop_vars is not None:
        # Remove specified unimportant variables
        data = data.drop(columns=drop_vars, errors="ignore")
        vars_info = vars_info[~vars_info["name"].isin(drop_vars)].reset_index()

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
    mice_kwargs = {key: value for key, value in kwargs.items() if key not in kernel_kwargs.keys()}

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


if __name__ == "__main__":
    # Create RNG for reproducibility
    rng = np.random.default_rng(325235)

    # Hardcoded paths, change to apply to new data
    data_path = Path("data/raw/ESA1821_neu_SO1.dta")

    save_folder = "data/preprocessed"
    data_save_path = Path(save_folder, "esa_2018_2021.pkl")
    data_imputed_save_path = Path(save_folder, "esa_2018_2021_imputed.pkl")
    imputation_kernel_save_path = Path(save_folder, "esa_2018_2021_imputation_kernel.pkl")
    vars_info_save_path = Path(save_folder, "esa_vars_info.pkl")

    allow_overwrite = True

    # Read in data
    data_reader = pd.read_stata(data_path, iterator=True, convert_categoricals=False, convert_missing=False)
    value_labels = data_reader.value_labels()
    var_labels = data_reader.variable_labels()
    with data_reader:
        data: pd.DataFrame = data_reader.read()  # type: ignore

    # Create information about variables
    vars_info = create_vars_info(var_labels, value_labels)

    # misc_vars = ["wicht", "wichtn", "audit_alt", "sds_c", "sds_k", "sds_a", "sds", "am_12"]
    misc_vars = ["wichtn", "sds_c", "sds_k", "sds_a"]

    general_vars = ["fall", "zeit", "stadt_bl", "int_art"]

    # socioecon_vars = [
    #     "alter", "ges", "hh", "hh18", "hh14", "allein", "staat", "Beruf", "erwerb", "fam", "schule", "hne", "isced",
    #     "westost_2", "hne_eq"
    # ]  # fmt: skip
    socioecon_vars = [
        "alter", "ges", "hh", "hh14", "allein", "staat", "Beruf", "erwerb", "fam", "schule", "hne", "westost_2",
    ]  # fmt: skip

    # alcohol_vars = [
    #     "alk30", "spir30", "mish30", "wein30gr", "mish30gr", "bier30gr", "spir30gr", "alk30gr", "alk12j", "alk12gr",
    #     "alkgr", "alkjahr", "alk30kat", "alkkat", "alkkons", "alkkon", "binge", "bierkons", "spirkons", "mishkons",
    #     "hazar", "harmf", "alkrisk", "alk12", "alk12m", "dsm_al", "dsm_al_m", "alk12mon", "alk12tag", "alpo30",
    #     "alpo_tag", "alpo30gr", "alpo12j", "alpo_tax", "alpo12gr", "alpogr", "alk_kat", "alpoli", "alpokons",
    #     "Miss_Abh", "vx210_2",
    # ]  # fmt: skip
    alcohol_vars = [
        "wein30gr", "mish30gr", "bier30gr", "spir30gr", "alk30gr", "alk30kat", "alkkons", "alkkon", "binge", "bierkons",
        "spirkons", "mishkons", "hazar", "harmf", "alkrisk", "dsm_al", "dsm_al_m", "alk_kat", "vx210_2",
    ]  # fmt: skip

    # nicotine_vars = [
    #     "akt_rau", "typ", "Shi", "Anzahl", "fag_sch", "ezp30", "typ_ezig", "dsm_ta", "alter_ezig", "psecdi_sch",
    #     "alter_hnb", "hnb30", "hnb_fag_sch", "typ_ezign", "typ_hnb", "typ_shish",
    # ]  # fmt: skip
    nicotine_vars = [
        "akt_rau", "typ", "Anzahl", "fag_sch", "ezp30", "typ_ezig", "dsm_ta", "psecdi_sch", "hnb30", "hnb_fag_sch",
        "typ_hnb", "typ_shish",
    ]  # fmt: skip

    # other_drugs_vars = [
    #     "andop_l", "hasch_12", "ecsta_12", "lsd_12", "heroi_12", "andop_12", "kokai_12", "crack_12", "pilz_12",
    #     "nps_12", "hasch_30", "andop_30", "schnu_30", "opiat_30", "spice_12", "amp_l", "amp_12", "meth_12", "gamp_12",
    #     "amp_30", "fr12_he", "fr12_ha1", "fr12_au1", "fr12_ec1", "fr12_ls1", "fr12_ao1", "fr12_ko1", "fr12_cr1",
    #     "fr12_sc1", "fr12_pi1", "fr12_nps1", "fr12_amp1", "fr12_meth1", "ls_alt", "dsm_ha", "dsm_au", "dsm_kc",
    #     "dsm_ha_m", "dsm_au_m", "dsm_kc_m", "fr12_spi1", "Miss_Abh_ha", "Miss_Abh_kc", "Miss_Abh_am", "Miss_Abh_dro",
    # ]  # fmt: skip
    other_drugs_vars = [
        "hasch_12", "ecsta_12", "lsd_12", "heroi_12", "andop_12", "kokai_12", "crack_12", "pilz_12", "nps_12", "amp_12",
        "meth_12", "fr12_he", "fr12_ha1", "fr12_au1", "fr12_ec1", "fr12_ls1", "fr12_ao1", "fr12_ko1", "fr12_cr1",
        "fr12_sc1", "fr12_pi1", "fr12_nps1", "fr12_meth1", "dsm_ha", "dsm_au", "dsm_kc", "dsm_ha_m", "dsm_au_m",
        "dsm_kc_m", "Miss_Abh_ha", "Miss_Abh_kc", "Miss_Abh_am", "Miss_Abh_dro",
    ]  # fmt: skip

    # medication_vars = [
    #     "schm_30", "schl_30", "ber_30", "anr_30", "app_30", "depr_30", "neur_30", "anab_30", "m_schm", "m_schl",
    #     "m_ber", "m_anr", "m_app", "m_depr", "m_neur", "m_anab", "t_schm", "t_schl", "t_ber", "t_anr", "t_app",
    #     "t_depr", "t_neur", "t_anab", "anr_12", "app_12", "anab_12", "dsm_schm", "dsm_schm_m", "Miss_Abh_schm",
    #     "dsm_med", "Miss_Abh_med", "rist", "dsm_schl", "dsm_ber", "dsm_schl_m", "Miss_Abh_schl", "dsm_ber_m",
    #     "Miss_Abh_ber", "anr_arzt", "app_arzt", "anab_arzt", "schmo_30", "schmno_30", "m_schmo", "m_schmno",
    #     "t_schmo", "t_schmno", "pomi", "rist_noa",
    # ]  # fmt: skip
    medication_vars = [
        "schm_30", "schl_30", "ber_30", "anr_30", "app_30", "depr_30", "neur_30", "anab_30", "m_schm", "m_schl",
        "m_ber", "m_anr", "m_app", "m_depr", "m_neur", "m_anab", "t_schm", "t_schl", "t_ber", "t_anr", "t_app",
        "t_depr", "t_neur", "t_anab", "anr_12", "app_12", "anab_12", "dsm_schm", "dsm_schm_m", "rist", "dsm_schl",
        "dsm_ber", "dsm_schl_m", "dsm_ber_m", "pomi", "rist_noa",
    ]  # fmt: skip

    # Assign group names
    groups_by_name = {
        "misc": misc_vars, "general": general_vars, "socioecon": socioecon_vars, "alcohol": alcohol_vars,
        "nicotine": nicotine_vars, "other_drugs": other_drugs_vars, "medication": medication_vars,
    }  # fmt: skip

    # Define variable types
    all_na_vars = [var for var in data.columns if data[var].isna().all()]

    # continuous_vars = [
    #     "wicht", "wichtn", "alter", "hh", "hh18", "hh14", "alk30", "spir30", "mish30", "wein30gr", "mish30gr",
    #     "alter_hnb", "bier30gr", "spir30gr", "alk30gr", "alk12j", "alk12gr", "alkgr", "alkjahr", "Anzahl", "alter_ezig",
    #     "pomi", "hne_eq",
    # ]  # fmt: skip
    continuous_vars = [
        "wichtn", "alter", "hh", "hh14", "wein30gr", "mish30gr", "bier30gr", "spir30gr", "alk30gr", "Anzahl", "pomi",
    ]  # fmt: skip

    # ordinal_vars = [
    #     "hne", "isced", "alk30kat", "alkkons", "alkkon", "binge", "alkrisk", "alk_kat", "akt_rau", "fag_sch",
    #     "psecdi_sch", "hnb_fag_sch", "fr12_he", "fr12_ha1", "fr12_au1", "fr12_ec1", "fr12_ls1", "fr12_ao1", "fr12_ko1",
    #     "fr12_cr1", "fr12_sc1", "fr12_pi1", "fr12_nps1", "fr12_meth1", "sds_c", "sds_k", "sds_a",
    # ]  # fmt: skip
    ordinal_vars = [
        "hne", "alk30kat", "alkkons", "alkkon", "binge", "alkrisk", "alk_kat", "akt_rau", "fag_sch", "psecdi_sch",
        "hnb_fag_sch", "fr12_he", "fr12_ha1", "fr12_au1", "fr12_ec1", "fr12_ls1", "fr12_ao1", "fr12_ko1", "fr12_cr1",
        "fr12_sc1", "fr12_pi1", "fr12_nps1", "fr12_meth1", "sds_c", "sds_k", "sds_a",
    ]  # fmt: skip

    binary_vars = [
        var
        for var in data.columns
        if var not in itertools.chain(all_na_vars, ordinal_vars, continuous_vars)
        and data[var].nunique(dropna=True) == 2
        # and set(value_labels.get(var, {}).values()) in [set(["nein", "ja"]), set(["Nein", "Ja"])]
    ]

    nominal_vars = [
        var
        for var in data.columns
        if var not in itertools.chain(all_na_vars, ordinal_vars, continuous_vars, binary_vars)
    ]

    # Assign type names
    types_by_name = {
        "all_na": all_na_vars, "continuous": continuous_vars, "binary": binary_vars, "nominal": nominal_vars,
        "ordinal": ordinal_vars,
    }  # fmt: skip

    # Add group and type info to information about variables
    vars_info = sort_groups_vars_info(vars_info, groups_by_name)
    vars_info = sort_types_vars_info(vars_info, types_by_name)

    # Preprocess data
    data, vars_info = preprocess_data(data, vars_info, remove_all_na_vars=True, drop_vars=["fall", "fr12_he"])
    data = data.assign(
        **{
            # Remove invalid value of akt_rau that occurs only once
            "akt_rau": lambda x: x["akt_rau"].cat.remove_categories(
                x["akt_rau"].value_counts().loc[lambda x: x == 1].index
            ),
            # Retain the 10 most common cities and lump the others into one category
            # (more than 10 are retained if there are ties)
            "stadt_bl": lambda x: (
                x["stadt_bl"]
                .map(
                    lambda val: val if val in x["stadt_bl"].value_counts().nlargest(n=10, keep="all").index else "other"
                )
                .astype("category")
            ),
            # Reorder the levels of alkrisk
            "alkrisk": lambda x: x["alkrisk"].cat.reorder_categories([1, 2, 3, 6, 4, 5]),
        }
    )

    imputation_kernel = impute_data(data, rng=rng, mi_number=3, mice_iterations=3, verbose=True)
    # Save the first of the imputed datasets
    data_imputed = typing.cast(pd.DataFrame, imputation_kernel.complete_data(dataset=0, inplace=False))

    # Save data with missing values, the imputation kernel, and information about variables
    save_data(data, data_save_path, label="dataset with missing values")
    save_data(data_imputed, data_imputed_save_path, label="imputed data")
    save_data(imputation_kernel, imputation_kernel_save_path, label="imputation kernel")
    save_data(vars_info, vars_info_save_path, label="information about variables")
