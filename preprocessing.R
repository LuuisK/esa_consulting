# Script for preprocessing.

# Load required packages and functions from funcs.R
library(haven)
library(dplyr)
library(forcats)
library(mice)

source("funcs.R")


# # Set a random seed for reproducibility
# set.seed(32523)

# Hardcoded paths, change to apply to new data
data_path <- file.path("data", "raw", "ESA1821_neu_SO1.dta")

save_folder <- file.path("data", "preprocessed")
data_save_path <- file.path(save_folder, "esa_2018_2021.pkl")
data_imputed_save_path <- file.path(save_folder, "esa_2018_2021_imputed.pkl")
vars_info_save_path <- file.path(save_folder, "esa_vars_info.pkl")

# Read in data
data <- read_dta(data_path)

# Create data frame containing information about variables
vars_info <- create_vars_info(data)

# Define variable groups
# fmt: skip
# misc_vars <- c(
#   "wicht", "wichtn", "audit_alt", "sds_c", "sds_k", "sds_a", "sds", "am_12"
# )
misc_vars <- c("wichtn", "sds_c", "sds_k", "sds_a")

general_vars <- c("fall", "zeit", "stadt_bl", "int_art")

# fmt: skip
# socioecon_vars <- c(
#   "alter", "ges", "hh", "hh18", "hh14", "allein", "staat", "Beruf", "erwerb",
#   "fam", "schule", "hne", "isced", "westost_2", "hne_eq"
# )
socioecon_vars <- c(
  "alter", "ges", "hh", "hh14", "allein", "staat", "Beruf", "erwerb", "fam",
  "schule", "hne", "westost_2"
)

# fmt: skip
# alcohol_vars <- c(
#   "alk30", "spir30", "mish30", "wein30gr", "mish30gr", "bier30gr", "spir30gr",
#   "alk30gr", "alk12j", "alk12gr", "alkgr", "alkjahr", "alk30kat", "alkkat",
#   "alkkons", "alkkon", "binge", "bierkons", "spirkons", "mishkons", "hazar",
#   "harmf", "alkrisk", "alk12", "alk12m", "dsm_al", "dsm_al_m", "alk12mon",
#   "alk12tag", "alpo30", "alpo_tag", "alpo30gr", "alpo12j", "alpo_tax",
#   "alpo12gr", "alpogr", "alk_kat", "alpoli", "alpokons", "Miss_Abh", "vx210_2"
# )
alcohol_vars <- c(
  "wein30gr", "mish30gr", "bier30gr", "spir30gr", "alk30gr", "alk30kat",
  "alkkons", "alkkon", "binge", "bierkons", "spirkons", "mishkons", "hazar",
  "harmf", "alkrisk", "dsm_al", "dsm_al_m", "alk_kat", "vx210_2"
)

# fmt: skip
# nicotine_vars <- c(
#   "akt_rau", "typ", "Shi", "Anzahl", "fag_sch", "ezp30", "typ_ezig", "dsm_ta",
#   "alter_ezig", "psecdi_sch", "alter_hnb", "hnb30", "hnb_fag_sch",
#   "typ_ezign", "typ_hnb", "typ_shish"
# )
nicotine_vars <- c(
  "akt_rau", "typ", "Anzahl", "fag_sch", "ezp30", "typ_ezig", "dsm_ta",
  "psecdi_sch", "hnb30", "hnb_fag_sch", "typ_hnb", "typ_shish"
)

# fmt: skip
# other_drugs_vars <- c(
#   "andop_l", "hasch_12", "ecsta_12", "lsd_12", "heroi_12", "andop_12",
#   "kokai_12", "crack_12", "pilz_12", "nps_12", "hasch_30", "andop_30",
#   "schnu_30", "opiat_30", "spice_12", "amp_l", "amp_12", "meth_12", "gamp_12",
#   "amp_30", "fr12_he", "fr12_ha1", "fr12_au1", "fr12_ec1", "fr12_ls1",
#   "fr12_ao1", "fr12_ko1", "fr12_cr1", "fr12_sc1", "fr12_pi1", "fr12_nps1",
#   "fr12_amp1", "fr12_meth1", "ls_alt", "dsm_ha", "dsm_au", "dsm_kc",
#   "dsm_ha_m", "dsm_au_m", "dsm_kc_m", "fr12_spi1", "Miss_Abh_ha",
#   "Miss_Abh_kc", "Miss_Abh_am", "Miss_Abh_dro"
# )
other_drugs_vars <- c(
  "hasch_12", "ecsta_12", "lsd_12", "heroi_12", "andop_12", "kokai_12",
  "crack_12", "pilz_12", "nps_12", "amp_12", "meth_12", "fr12_he", "fr12_ha1",
  "fr12_au1", "fr12_ec1", "fr12_ls1", "fr12_ao1", "fr12_ko1", "fr12_cr1",
  "fr12_sc1", "fr12_pi1", "fr12_nps1", "fr12_meth1", "dsm_ha", "dsm_au",
  "dsm_kc", "dsm_ha_m", "dsm_au_m", "dsm_kc_m", "Miss_Abh_ha", "Miss_Abh_kc",
  "Miss_Abh_am", "Miss_Abh_dro"
)

# fmt: skip
# medication_vars <- c(
#   "schm_30", "schl_30", "ber_30", "anr_30", "app_30", "depr_30", "neur_30",
#   "anab_30", "m_schm", "m_schl", "m_ber", "m_anr", "m_app", "m_depr",
#   "m_neur", "m_anab", "t_schm", "t_schl", "t_ber", "t_anr", "t_app", "t_depr",
#   "t_neur", "t_anab", "anr_12", "app_12", "anab_12", "dsm_schm", "dsm_schm_m",
#   "Miss_Abh_schm", "dsm_med", "Miss_Abh_med", "rist", "dsm_schl", "dsm_ber",
#   "dsm_schl_m", "Miss_Abh_schl", "dsm_ber_m", "Miss_Abh_ber", "anr_arzt",
#   "app_arzt", "anab_arzt", "schmo_30", "schmno_30", "m_schmo", "m_schmno",
#   "t_schmo", "t_schmno", "pomi", "rist_noa"
# )
medication_vars <- c(
  "schm_30", "schl_30", "ber_30", "anr_30", "app_30", "depr_30", "neur_30",
  "anab_30", "m_schm", "m_schl", "m_ber", "m_anr", "m_app", "m_depr", "m_neur",
  "m_anab", "t_schm", "t_schl", "t_ber", "t_anr", "t_app", "t_depr", "t_neur",
  "t_anab", "anr_12", "app_12", "anab_12", "dsm_schm", "dsm_schm_m", "rist",
  "dsm_schl", "dsm_ber", "dsm_schl_m", "dsm_ber_m", "pomi", "rist_noa"
)

# Assign group names
groups_by_name <- list(
  misc = misc_vars,
  general = general_vars,
  socioecon = socioecon_vars,
  alcohol = alcohol_vars,
  nicotine = nicotine_vars,
  other_drugs = other_drugs_vars,
  medication = medication_vars
)

# Define variable types
all_na_vars <- data %>%
  vapply(function(x) all(is.na(x)), FUN.VALUE = logical(1)) %>%
  which %>%
  names

# fmt: skip
# continuous_vars <- c(
#   "wicht", "wichtn", "alter", "hh", "hh18", "hh14", "alk30", "spir30",
#   "mish30", "wein30gr", "mish30gr", "alter_hnb", "bier30gr", "spir30gr",
#   "alk30gr", "alk12j", "alk12gr", "alkgr", "alkjahr", "Anzahl", "alter_ezig",
#   "pomi", "hne_eq"
# )
continuous_vars <- c(
  "wichtn", "alter", "hh", "hh14", "wein30gr", "mish30gr", "bier30gr",
  "spir30gr", "alk30gr", "Anzahl", "pomi"
)

# fmt: skip
# ordinal_vars <- c(
#   "hne", "isced", "alk30kat", "alkkons", "alkkon", "binge", "alkrisk",
#   "alk_kat", "akt_rau", "fag_sch", "psecdi_sch", "hnb_fag_sch", "fr12_he",
#   "fr12_ha1", "fr12_au1", "fr12_ec1", "fr12_ls1", "fr12_ao1", "fr12_ko1",
#   "fr12_cr1", "fr12_sc1", "fr12_pi1", "fr12_nps1", "fr12_meth1", "sds_c",
#   "sds_k", "sds_a"
# )
ordinal_vars <- c(
  "hne", "alk30kat", "alkkons", "alkkon", "binge", "alkrisk", "alk_kat",
  "akt_rau", "fag_sch", "psecdi_sch", "hnb_fag_sch", "fr12_he", "fr12_ha1",
  "fr12_au1", "fr12_ec1", "fr12_ls1", "fr12_ao1", "fr12_ko1", "fr12_cr1",
  "fr12_sc1", "fr12_pi1", "fr12_nps1", "fr12_meth1", "sds_c", "sds_k", "sds_a"
)

nominal_vars <- (function(vars, drop_vars) vars[vars %not_in% drop_vars])(
  names(data),
  c(all_na_vars, ordinal_vars, continuous_vars)
)

binary_vars <- nominal_vars %>%
  vapply(
    function(x) sum(!is.na(unique(data[x]))) == 2,
    FUN.VALUE = logical(1)
  ) %>%
  # vapply(
  #   function(x) setequal(names(attr(data[x], "labels")), c("nein", "ja")),
  #   FUN.VALUE = logical(1)
  # ) %>%
  which %>%
  names

nominal_vars <- (function(vars, drop_vars) vars[vars %not_in% drop_vars])(
  nominal_vars,
  binary_vars
)

# Assign type names
types_by_name <- list(
  all_na = all_na_vars,
  continuous = continuous_vars,
  binary = binary_vars,
  nominal = nominal_vars,
  ordinal = ordinal_vars
)

# Add group and type info to information about variables
vars_info <- sort_groups_vars_info(vars_info, groups_by_name)
vars_info <- sort_types_vars_info(vars_info, types_by_name)

# Preprocess data
output <- preprocess_data(
  data,
  vars_info,
  remove_all_na_vars = TRUE,
  drop_vars = c("fall")
)
data <- output[["data"]]
vars_info <- output[["vars_info"]]
data <- data %>%
  mutate(
    # Remove invalid value of akt_rau that occurs only once
    akt_rau = akt_rau %>%
      fct_lump_min(min = 2, other_level = NA) %>%
      fct_na_level_to_value,
    #' Retain the 10 most common cities and lump the others into one category
    #' (more than 10 are retained if there are ties)
    stadt_bl = stadt_bl %>%
      fct_lump_n(n = 10, other_level = "other", ties.method = "min"),
    # # Create discrete alter_ezig bins
    # alter_ezig = alter_ezig %>% cut_number(n = 4),
    # Reorder the levels of alkrisk
    alkrisk = fct_relevel(alkrisk, c("1", "2", "3", "6", "4", "5"))
  )

#' Multiply impute data with mice
#' (nnet.MaxNWts may need to be increased to train a bigger neural net)
#'
#' TODO: We need to reduce the data used for imputation,
#' see: https://stefvanbuuren.name/fimd/sec-modelform.html
data_imputed <- mice(
  data,
  m = 5,
  maxit = 5,
  seed = 9876,
  #' Use polyreg instead of equivalent logreg for binary data because of
  #' convergence issues due to the maxit argument of logreg,
  #' see: https://stackoverflow.com/questions/58831710/
  #' mice-glm-fit-algorithm-did-not-converge/60056364#60056364
  defaultMethod = c("pmm", "polyreg", "polyreg", "polr"),
  # #' Not removing collinearity can destroy some correlation,
  # #' see: https://stefvanbuuren.name/fimd/sec-modelform.html
  # remove.collinear = FALSE,
  nnet.MaxNWts = 10000
)

# Save data with missing values, imputed data, and information about variables
save_data(data, data_save_path)
save_data(data_imputed, data_imputed_save_path)
save_data(vars_info, vars_info_save_path)
