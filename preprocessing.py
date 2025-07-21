import itertools
import os

import numpy as np
import pandas as pd

from utils import check_save_path

# Hardcoded paths, change to apply to new data
data_folder = "data/readin"
save_folder = "data/preprocessed"
dataset_filename = "esa_2018.csv"
vars_info_filename = "esa_vars_info.csv"
data_path = os.path.join(data_folder, dataset_filename)
var_info_path = os.path.join(data_folder, vars_info_filename)

allow_overwrite = False


def group_vars(vars_info: pd.DataFrame) -> pd.DataFrame:
    # Define variable groups
    misc_vars = ["wicht", "wichtn", "audit_alt", "sds_c", "sds_k", "sds_a", "sds", "am_12"]
    general_vars = ["zeit", "stadt_bl", "int_art"]  # ["fall", "zeit", "stadt_bl", "int_art"]
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

    # Check that there are no multiply assigned variables
    all_grouped_vars = list(itertools.chain.from_iterable(groups_by_name.values()))
    duplicates = set([var for var in all_grouped_vars if all_grouped_vars.count(var) > 1])
    if duplicates:
        raise ValueError(f"Some variables have multiple group membership: {duplicates}.")

    # Assign groups to variables
    vars_info["group"] = pd.NA
    for group_name, group_vars in groups_by_name.items():
        vars_info.loc[vars_info.index.isin(group_vars), "group"] = group_name

    # Check that every variable is assigned to a group
    not_assigned = vars_info.drop("fall")[vars_info.drop("fall")["group"].isna()]
    if not_assigned.shape[0] > 0:
        raise ValueError(f"Some variables are not assigned to a group: {not_assigned.index.tolist()}.")

    # Return grouped variables
    return vars_info


def sort_vars(vars_info: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    # Define variable types
    all_na = [var for var, val in data.items() if val.isna().all()]

    continuous = [
        "wicht", "wichtn", "wein30gr", "mish30gr", "bier30gr", "spir30gr", "alk30gr", "alk12gr", "alkgr", "alkjahr",
        "hne_eq",
    ]  # fmt: skip

    need_preprocessing = [
        "alter", "hh", "hh18", "alk30", "spir30", "mish30", "alk12j", "Anzahl", "alter_ezig",
    ]  # fmt: skip

    # yes_no_nan = (
    #     [
    #         var for var, val in data.items()
    #         if len(levels := val.unique()) in [2, 3]
    #         and all(a in ["ja", "nein"] or a is np.nan for a in levels)
    #     ]
    # )

    categorical = [var for var in data.columns if var not in itertools.chain(all_na, continuous, need_preprocessing)]

    # Assign type names
    types_by_name = {
        "all_na": all_na, "continuous": continuous, "categorical": categorical,
    }  # fmt: skip

    # Check that there are no multiply assigned variables
    all_typed_vars = list(itertools.chain.from_iterable(types_by_name.values()))
    duplicates = set([var for var in all_typed_vars if all_typed_vars.count(var) > 1])
    if duplicates:
        raise ValueError(f"Some variables have several types: {duplicates}.")

    # Assign types to variables
    vars_info["type"] = pd.NA
    for type_name, type_vars in types_by_name.items():
        vars_info.loc[vars_info.index.isin(type_vars), "type"] = type_name

    # See which variables are not assigned a type
    not_assigned = vars_info.drop("fall")[vars_info.drop("fall")["type"].isna()]
    if not_assigned.shape[0] > 0:
        print(f"Some variables are not assigned a type: {not_assigned.index.tolist()}.")

    # Return variable information
    return vars_info


def preprocess_vars_info(vars_info: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses information about variables:
    - Partitions variables into groups.
    - Sorts variables into types: all_na, categorical, continuous, or NA in case of no type assignment.
    """
    # Partition variables into groups
    vars_info = group_vars(vars_info)

    # Specify types of variables
    vars_info = sort_vars(vars_info, data)

    # Return preprocessed variables
    return vars_info


def preprocess_data(data: pd.DataFrame, vars_info: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data:
    - Drops respondents with only missing answers.
    - Encodes factor levels as categorical codes (0, 1, 2, ...) and saves the original levels in vars_info.
    - Removes unimportant features.
    """
    # Convert categorical variables to factors
    categorical = vars_info.index[vars_info["type"] == "categorical"]
    data[categorical] = data[categorical].apply(lambda x: pd.Categorical(x), axis=0)

    # Save the levels that category codes stand for
    def get_levels(var):
        if var not in categorical:
            return pd.NA
        levels = data[var].cat.categories.to_list()
        if np.any(data[var].cat.codes < 0):
            levels.append(pd.NA)
        return levels

    vars_info["levels"] = pd.Series(vars_info.index.map(get_levels), vars_info.index)

    # Replace the levels by category codes
    def get_codes(x):
        num_cats = len(x.cat.categories)
        codes = x.cat.codes
        return codes.replace(-1, num_cats)

    data[categorical] = data[categorical].apply(get_codes)

    # # Remove unimportant features
    # data = data.drop(columns="stadt_bl")

    # Return preprocessed data
    return data


if __name__ == "__main__":
    # Load data and information about variables
    data_folder, data_filename = os.path.split(data_path)
    data_name, data_extension = os.path.splitext(data_filename)

    vars_info = pd.read_csv(var_info_path, index_col=0, low_memory=False)
    data = pd.read_csv(data_path, index_col=0, low_memory=False)

    # Preprocess information about variables
    vars_info = preprocess_vars_info(vars_info, data)

    # Preprocess data
    data = preprocess_data(data, vars_info)

    # Save information about grouped variables
    save_path = os.path.join(save_folder, "esa_grouped_vars_info.csv")
    check_save_path(save_path, allow_overwrite=allow_overwrite)
    vars_info.to_csv(save_path)
    print(f"Saved grouped information about variables as CSV to {save_path}.")

    # Save preprocessed data
    save_path = os.path.join(save_folder, dataset_filename)
    check_save_path(save_path, allow_overwrite=allow_overwrite)
    data.to_csv(save_path)
    print(f"Saved preprocessed dataset as CSV to {save_path}.")
