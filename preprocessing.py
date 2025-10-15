"""
Script for preprocessing (including imputation).

Run scripts as interactive notebooks in the follwing order:
    preprocessing.py  ->  plot_corr_assoc.py  ->  dimensionality_reduction.py  ->  clustering.py
"""

import itertools
import os
import warnings

import miceforest as mf
import numpy as np
import pandas as pd

from utils import check_save_path

pd.options.mode.copy_on_write = True


# Hardcoded paths, change to apply to new data
data_folder = os.path.join("data", "raw")
data_path = os.path.join(data_folder, "ESA1821_neu_SO.dta")

save_folder = os.path.join("data", "preprocessed")
data_save_path = os.path.join(save_folder, "esa_2018_2021.pkl")
data_imputed_save_path = os.path.join(save_folder, "esa_2018_2021_imputed.pkl")
vars_info_save_path = os.path.join(save_folder, "esa_vars_info.pkl")

allow_overwrite = True


def get_var_groups() -> dict[str, list[str]]:
    """
    Specifies named groups of variables. Change to apply to new data.
    """
    # Define variable groups
    misc_vars = ["wicht", "wichtn", "audit_alt", "sds_c", "sds_k", "sds_a", "sds", "am_12"]
    general_vars = ["fall", "zeit", "stadt_bl", "int_art"]
    socioecon_vars = [
        "alter", "ges", "hh", "hh18", "hh14", "allein", "staat", "Beruf", "erwerb", "fam", "schule", "hne", "isced",
        "westost_2", "hne_eq",
    ]  # fmt: skip
    alcohol_vars = [
        "alk30", "spir30", "mish30", "wein30gr", "mish30gr", "bier30gr", "spir30gr", "alk30gr", "alk12j", "alk12gr",
        "alkgr", "alkjahr", "alk30kat", "alkkat", "alkkons", "alkkon", "binge", "bierkons", "spirkons", "mishkons",
        "hazar", "harmf", "alkrisk", "alk12", "alk12m", "dsm_al", "dsm_al_m", "alk12mon", "alk12tag", "alpo30",
        "alpo_tag", "alpo30gr", "alpo12j", "alpo_tax", "alpo12gr", "alpogr", "alk_kat", "alpoli", "alpokons",
        "Miss_Abh", "vx210_2",
    ]  # fmt: skip
    nicotine_vars = [
        "akt_rau", "typ", "Shi", "Anzahl", "fag_sch", "ezp30", "typ_ezig", "dsm_ta", "alter_ezig", "psecdi_sch",
        "alter_hnb", "hnb30", "hnb_fag_sch", "typ_ezign", "typ_hnb", "typ_shish",
    ]  # fmt: skip
    other_drugs_vars = [
        "andop_l", "hasch_12", "ecsta_12", "lsd_12", "heroi_12", "andop_12", "kokai_12", "crack_12", "pilz_12",
        "nps_12", "hasch_30", "andop_30", "schnu_30", "opiat_30", "spice_12", "amp_l", "amp_12", "meth_12", "gamp_12",
        "amp_30", "fr12_he", "fr12_ha1", "fr12_au1", "fr12_ec1", "fr12_ls1", "fr12_ao1", "fr12_ko1", "fr12_cr1",
        "fr12_sc1", "fr12_pi1", "fr12_nps1", "fr12_amp1", "fr12_meth1", "ls_alt", "dsm_ha", "dsm_au", "dsm_kc",
        "dsm_ha_m", "dsm_au_m", "dsm_kc_m", "fr12_spi1", "Miss_Abh_ha", "Miss_Abh_kc", "Miss_Abh_am", "Miss_Abh_dro",
    ]  # fmt: skip
    medication_vars = [
        "schm_30", "schl_30", "ber_30", "anr_30", "app_30", "depr_30", "neur_30", "anab_30", "m_schm", "m_schl",
        "m_ber", "m_anr", "m_app", "m_depr", "m_neur", "m_anab", "t_schm", "t_schl", "t_ber", "t_anr", "t_app",
        "t_depr", "t_neur", "t_anab", "anr_12", "app_12", "anab_12", "dsm_schm", "dsm_schm_m", "Miss_Abh_schm",
        "dsm_med", "Miss_Abh_med", "rist", "dsm_schl", "dsm_ber", "dsm_schl_m", "Miss_Abh_schl", "dsm_ber_m",
        "Miss_Abh_ber", "anr_arzt", "app_arzt", "anab_arzt", "schmo_30", "schmno_30", "m_schmo", "m_schmno", "t_schmo",
        "t_schmno", "pomi", "rist_noa",
    ]  # fmt: skip

    # Assign group names
    groups_by_name = {
        "misc": misc_vars, "general": general_vars, "socioecon": socioecon_vars, "alcohol": alcohol_vars,
        "nicotine": nicotine_vars, "other_drugs": other_drugs_vars, "medication": medication_vars,
    }  # fmt: skip

    return groups_by_name


def get_var_types(data: pd.DataFrame) -> dict[str, list[str]]:
    """
    Specifies variable types. Change to apply to new data.
    """
    # Define variable types
    all_na_vars = [var for var in data.columns if data[var].isna().all()]

    continuous_vars = [
        "wicht", "wichtn", "wein30gr", "mish30gr", "bier30gr", "spir30gr", "alk30gr", "alk12gr", "alkgr", "alkjahr",
        "hne_eq",
    ]  # fmt: skip

    need_preprocessing_vars = [
        "alter", "hh", "hh18", "alk30", "spir30", "mish30", "alk12j", "Anzahl", "alter_ezig",
    ]  # fmt: skip

    # yes_no_nan_vars = (
    #     [
    #         var for var in data.columns
    #         if data[var].nunique() == 2 and set(["nein", "ja"]) == set(value_labels.get(var, {}).values())
    #     ]
    # )

    categorical_vars = [
        var for var in data.columns if var not in itertools.chain(all_na_vars, continuous_vars, need_preprocessing_vars)
    ]

    # Assign type names
    types_by_name = {
        "all_na": all_na_vars, "continuous": continuous_vars, "categorical": categorical_vars,
    }  # fmt: skip

    return types_by_name


def create_vars_info(var_labels: dict[str, str], value_labels: dict[str, dict[float, str]]) -> pd.DataFrame:
    """
    Creates a dataframe containing information about variables.
    """
    vars_info = pd.DataFrame(var_labels.items(), columns=["variable", "description"], dtype=pd.StringDtype())
    vars_info["labels"] = pd.NA
    for var in vars_info["variable"]:
        if var in value_labels.keys():
            vars_info.loc[vars_info["variable"] == var, "labels"] = [value_labels[var]]
        else:
            vars_info.loc[vars_info["variable"] == var, "labels"] = [{}]
    return vars_info


def sort_groups_vars_info(vars_info: pd.DataFrame, groups_by_name: dict[str, list[str]]) -> pd.DataFrame:
    """
    Partitions variables in vars_info into groups.
    """
    # Check that there are no multiply assigned variables
    all_grouped_vars = list(itertools.chain.from_iterable(groups_by_name.values()))
    duplicates = set([var for var in all_grouped_vars if all_grouped_vars.count(var) > 1])
    if duplicates:
        raise ValueError(f"Some variables have multiple group membership: {duplicates}.")

    # Assign groups to variables
    vars_info["group"] = pd.Categorical(
        values=len(vars_info) * [pd.NA], categories=pd.Series(groups_by_name.keys(), dtype=pd.StringDtype())
    )
    for group_name, group_vars in groups_by_name.items():
        vars_info.loc[vars_info["variable"].isin(group_vars), "group"] = group_name

    # Check that every variable is assigned to a group
    not_assigned = vars_info[vars_info["group"].isna()]
    if not_assigned.shape[0] > 0:
        raise ValueError(f"Some variables are not assigned to a group: {not_assigned['variable'].tolist()}.")

    return vars_info


def sort_types_vars_info(vars_info: pd.DataFrame, types_by_name: dict[str, list[str]]) -> pd.DataFrame:
    """
    Sorts variables in vars_info into types.

    The type of a variable is one of: all_na, categorical, continuous, or NA in case of no type assignment.
    """
    # Check that there are no multiply assigned variables
    all_typed_vars = list(itertools.chain.from_iterable(types_by_name.values()))
    duplicates = set([var for var in all_typed_vars if all_typed_vars.count(var) > 1])
    if duplicates:
        raise ValueError(f"Some variables have several types: {duplicates}.")

    # Assign types to variables
    vars_info["type"] = pd.Categorical(
        values=len(vars_info) * [pd.NA], categories=pd.Series(types_by_name.keys(), dtype=pd.StringDtype())
    )
    for type_name, type_vars in types_by_name.items():
        vars_info.loc[vars_info["variable"].isin(type_vars), "type"] = type_name

    # Display variables that are not assigned a type
    not_assigned = vars_info[vars_info["type"].isna()]
    if not_assigned.shape[0] > 0:
        print(f"Some variables are not assigned a type: {not_assigned['variable'].tolist()}.")

    return vars_info


def preprocess_vars_info(
    vars_info: pd.DataFrame, groups_by_name: dict[str, list[str]], types_by_name: dict[str, list[str]]
) -> pd.DataFrame:
    """
    Preprocesses information about variables:
    - Partitions variables into groups.
    - Sorts variables into types: all_na, categorical, continuous, or NA in case of no type assignment.
    """
    # Partition variables into groups
    vars_info = sort_groups_vars_info(vars_info, groups_by_name)

    # Specify types of variables
    vars_info = sort_types_vars_info(vars_info, types_by_name)

    return vars_info


def preprocess_data(
    data: pd.DataFrame, vars_info: pd.DataFrame, remove_all_na_vars: bool = True, drop_vars: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the data:
    - Converts factorial data to have categorical data type and saves the original levels in vars_info.
    - Removes unimportant features.
    """
    # Convert categorical variables to factors
    categorical_vars: pd.Series[pd.StringDtype] = vars_info.loc[vars_info["type"] == "categorical", "variable"]
    data[categorical_vars] = data[categorical_vars].astype("category")

    if remove_all_na_vars:
        # Remove features that are NA everywhere from the dataset and vars_info
        all_na_vars: pd.Series[pd.StringDtype] = vars_info.loc[vars_info["type"] == "all_na", "variable"]
        print(
            f"Dropped the following variables from the dataset because they are missing everywhere: {all_na_vars.to_list()}"
        )
        data = data.drop(columns=all_na_vars)
        vars_info = vars_info[vars_info["type"] != "all_na"]
        vars_info["type"] = vars_info["type"].cat.remove_categories("all_na")

    if drop_vars is not None:
        # Remove specified unimportant variables
        data = data.drop(columns=drop_vars, errors="ignore")
        vars_info = vars_info[~vars_info["variable"].isin(drop_vars)].reset_index()

    return data, vars_info


def impute_data(
    data: pd.DataFrame,
    rng: np.random.Generator | np.random.BitGenerator | None = None,
    mi_number: int = 1,
    mice_iterations: int = 3,
    **kwargs,
) -> pd.DataFrame:
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
    kernel = mf.ImputationKernel(
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
        kernel.mice(iterations=mice_iterations, **mice_kwargs)
    data_imputed: pd.DataFrame = kernel.complete_data()

    return data_imputed


if __name__ == "__main__":
    # Create RNG for reproducibility
    rng = np.random.default_rng(325235)

    # Load data and information about variables
    data_folder, data_filename = os.path.split(data_path)
    data_name, data_extension = os.path.splitext(data_filename)

    # Read in data
    data_reader = pd.read_stata(data_path, iterator=True, convert_categoricals=False, convert_missing=False)
    value_labels = data_reader.value_labels()
    var_labels = data_reader.variable_labels()
    with data_reader:
        data: pd.DataFrame = data_reader.read()

    # Create information about variables
    vars_info = create_vars_info(var_labels, value_labels)
    groups_by_name = get_var_groups()
    types_by_name = get_var_types(data=data)
    vars_info = preprocess_vars_info(vars_info, groups_by_name, types_by_name)

    # Preprocess data
    data, vars_info = preprocess_data(data, vars_info, remove_all_na_vars=True, drop_vars=["fall"])
    data_imputed = impute_data(data, rng=rng, mi_number=1, mice_iterations=2, verbose=True)

    # Save data with missing values, imputed data, and information about variables
    check_save_path(data_save_path, allow_overwrite=allow_overwrite)
    data.to_pickle(data_save_path)
    print(f"Saved dataset with missing values as Pickle to {data_save_path}.")

    check_save_path(data_imputed_save_path, allow_overwrite=allow_overwrite)
    data_imputed.to_pickle(data_imputed_save_path)
    print(f"Saved imputed data as Pickle to {data_imputed_save_path}.")

    check_save_path(vars_info_save_path, allow_overwrite=allow_overwrite)
    vars_info.to_pickle(vars_info_save_path)
    print(f"Saved information about variables as Pickle to {vars_info_save_path}.")
