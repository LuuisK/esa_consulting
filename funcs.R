#' Script defining functions.



# Load required packages
box::use(
  readr[write_rds],
  tools[file_ext],
  checkmate[assert_string, assert_data_frame],
  dplyr[`%>%`, mutate, filter, pull, all_of, across],
  ggplot2[cut_number],
  tidyr[replace_na],
  utils[tail],
  haven[as_factor]
)



#' "Not In" operator.
#'
#' @export
`%not_in%` = Negate(`%in%`)



#' Empty a folder.
#'
#' Empties the given folder by deleting every file and folder contained in it.
#'
#' @export
empty_folder <- function(folder) {
  # Check that the input folder is a string
  assert_string(folder)

  files <- list.files(
    folder,
    include.dirs = TRUE,
    full.names = TRUE,
    recursive = TRUE
  )
  file.remove(files)
  return(invisible(NULL))
}



#' Save a data frame as an rds-file.
#'
#' Saves the passed data with a given file name and prints it with the label.
#'
#' @export
save_data <- function(data, save_path, label="data") {
  # Check that the input data is a data frame and the file name is a string
  assert_data_frame(data)
  assert_string(save_path)

  ext <- file_ext(save_path)
  write_rds(data, file = save_path)
  sprintf("Saved %s as an %s file to %s.", label, ext, save_path)
}



#' Sort variables into groups.
#'
#' Returns the input data frame of information about variables with an added
#' grouping of the variables (miscellaneous, general, socioeconomic, alcohol,
#' nicotine, other drugs, medication).
#'
#' @export
sort_var_groups <- function(vars_info, data) {
  # Check that the inputs vars_info and data are data frames
  assert_data_frame(vars_info)
  assert_data_frame(data)

  # Define variable groups
  misc_vars <- c(
    "wicht", "wichtn", "audit_alt", "sds_c", "sds_k", "sds_a", "sds", "am_12"
  )
  general_vars <- c("fall", "zeit", "stadt_bl", "int_art")
  socioecon_vars <- c(
    "alter", "ges", "hh", "hh18", "hh14", "allein", "staat", "Beruf", "erwerb",
    "fam", "schule", "hne", "isced", "westost_2", "hne_eq"
  )
  alcohol_vars <- c(
    "alk30", "spir30", "mish30", "wein30gr", "mish30gr", "bier30gr", "spir30gr",
    "alk30gr", "alk12j", "alk12gr", "alkgr", "alkjahr", "alk30kat", "alkkat",
    "alkkons", "alkkon", "binge", "bierkons", "spirkons", "mishkons", "hazar",
    "harmf", "alkrisk", "alk12", "alk12m", "dsm_al", "dsm_al_m", "alk12mon",
    "alk12tag", "alpo30", "alpo_tag", "alpo30gr", "alpo12j", "alpo_tax",
    "alpo12gr", "alpogr", "alk_kat", "alpoli", "alpokons", "Miss_Abh",
    "vx210_2"
  )
  nicotine_vars <- c(
    "akt_rau", "typ", "Shi", "Anzahl", "fag_sch", "ezp30", "typ_ezig", "dsm_ta",
    "alter_ezig", "psecdi_sch", "alter_hnb", "hnb30", "hnb_fag_sch",
    "typ_ezign", "typ_hnb", "typ_shish"
  )
  other_drugs_vars <- c(
    "andop_l", "hasch_12", "ecsta_12", "lsd_12", "heroi_12", "andop_12",
    "kokai_12", "crack_12", "pilz_12", "nps_12", "hasch_30", "andop_30",
    "schnu_30", "opiat_30", "spice_12", "amp_l", "amp_12", "meth_12", "gamp_12",
    "amp_30", "fr12_he", "fr12_ha1", "fr12_au1", "fr12_ec1", "fr12_ls1",
    "fr12_ao1", "fr12_ko1", "fr12_cr1", "fr12_sc1", "fr12_pi1", "fr12_nps1",
    "fr12_amp1", "fr12_meth1", "ls_alt", "dsm_ha", "dsm_au", "dsm_kc",
    "dsm_ha_m", "dsm_au_m", "dsm_kc_m", "fr12_spi1", "Miss_Abh_ha",
    "Miss_Abh_kc", "Miss_Abh_am", "Miss_Abh_dro"
  )
  medication_vars <- c(
    "schm_30", "schl_30", "ber_30", "anr_30", "app_30", "depr_30", "neur_30",
    "anab_30", "m_schm", "m_schl", "m_ber", "m_anr", "m_app", "m_depr",
    "m_neur", "m_anab", "t_schm", "t_schl", "t_ber", "t_anr", "t_app", "t_depr",
    "t_neur", "t_anab", "anr_12", "app_12", "anab_12", "dsm_schm", "dsm_schm_m",
    "Miss_Abh_schm", "dsm_med", "Miss_Abh_med", "rist", "dsm_schl", "dsm_ber",
    "dsm_schl_m", "Miss_Abh_schl", "dsm_ber_m", "Miss_Abh_ber", "anr_arzt",
    "app_arzt", "anab_arzt", "schmo_30", "schmno_30", "m_schmo", "m_schmno",
    "t_schmo", "t_schmno", "pomi", "rist_noa"
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

  # Check that there are no multiply assigned variables
  all_grouped_vars <- unlist(groups_by_name, use.names = FALSE)
  if (any(table(all_grouped_vars) != 1)) {
    stop("Variables may not be assigned to more than one group.")
  }

  # See which variables are not assigned a group
  not_assigned <- setdiff(names(data), all_grouped_vars)
  if (length(not_assigned) > 0) {
    sprintf(
      "Some variables are not assigned to a group: %s.",
      do.call(paste0, list(not_assigned, collapse = ", "))
    )
  }

  # Assign groups to variables
  for (group_name in names(groups_by_name)) {
    group_vars <- groups_by_name[[group_name]]
    mask <- vars_info$name %in% group_vars
    vars_info[mask, "group"] <- group_name
  }

  # Return grouped variables
  return(vars_info)
}



#' Sort variables by type.
#'
#' Returns the input data frame of information about variables with an added
#' sorting by variable type (all_na, categorical, continuous, or no type (NA)).
#'
#' @export
sort_var_types <- function(vars_info, data) {
  # Check that the inputs vars_info and data are data frames
  assert_data_frame(vars_info)
  assert_data_frame(data)

  # Define variable types
  mask <- vapply(
    data, function(x) all(is.na(x)), FUN.VALUE = logical(1)
  )
  all_na_vars <- names(data)[mask]

  continuous_vars <- c(
    "wicht", "wichtn", "wein30gr", "mish30gr", "bier30gr", "spir30gr",
    "alk30gr", "alk12gr", "alkgr", "alkjahr", "hne_eq"
  )

  quasicontinuous_vars <- c(
    "alter", "hh", "hh18", "hh14", "hne", "alk30", "spir30", "mish30", "alk12j",
    "Anzahl", "alter_ezig", "alter_hnb"
  ) # The hne is ordinal with equal spacing except the borders (1 and 13).
  #' Ordinal but not quasicontinuous: c("isced", "alk30kat", "alkkons",
  #' "alkkon", "binge", "alk_kat", "akt_rau", "fag_sch", "psecdi_sch",
  #' "hnb_fag_sch", "fr12_he", "fr12_ha1", "fr12_au1", "fr12_ec1", "fr12_ls1",
  #' "fr12_ao1", "fr12_ko1", "fr12_cr1", "fr12_sc1", "fr12_pi1", "fr12_nps1",
  #' "fr12_meth1")
  #' Almost ordinal: c("erwerb", "schule")

  mask <- vapply(
    data,
    function(x) setequal(names(attr(x, "labels")), c("nein", "ja")),
    FUN.VALUE = logical(1)
  )
  binary_vars <- names(data)[mask]
  binary_vars <- binary_vars[binary_vars %not_in% all_na_vars]


  mask <- names(data) %not_in% c(
    all_na_vars, continuous_vars, quasicontinuous_vars, binary_vars
  )
  categorical_vars <- names(data)[mask]

  # Assign type names
  types_by_name <- list(
    all_na = all_na_vars,
    quasicontinuous = quasicontinuous_vars,
    continuous = c(continuous_vars, quasicontinuous_vars),
    binary = binary_vars,
    categorical = c(categorical_vars, binary_vars)
  )

  # See which variables are not assigned a type
  all_typed_vars <- unlist(types_by_name, use.names = FALSE)
  not_assigned <- setdiff(names(data), all_typed_vars)
  if (length(not_assigned) > 0) {
    sprintf(
      "Some variables are not assigned a type: %s.",
      do.call(paste0, list(not_assigned, collapse = ", "))
    )
  }

  # Assign types to variables
  for (type_name in names(types_by_name)) {
    type_vars <- types_by_name[[type_name]]
    vars_info <- vars_info %>% mutate("{type_name}" := name %in% type_vars)
  }
  # for (type_name in names(types_by_name)) {
  #   type_vars <- types_by_name[[type_name]]
  #   mask <- vars_info$name %in% type_vars
  #   vars_info[mask, "type"] <- type_name
  # }

  return(vars_info)
}



#' Correct errors and reformat data.
#'
#' Corrects inconsistencies and recodes variables in the input data frame
#' containing the ESA data.
#'
#' @export
correct_and_reformat <- function(data, vars_info) {
  # Check that the inputs data and vars_info are data frames
  assert_data_frame(data)
  assert_data_frame(vars_info)

  reorder_alkrisk_levels <- function(var) {
    labels <- attr(var, "labels")
    names(labels) <- c(names(labels)[1:3], names(labels)[6], names(labels)[4:5])
    attr(var, "labels") <- labels
    mask <- var %in% c(4, 5, 6)
    var <- replace(
      var, mask, ifelse(var[mask] == 6, var[mask] - 2, var[mask] + 1)
    )
    return(var)
  }
  data <- data %>%
    mutate(
      # remove invalid value of akt_rau
      akt_rau = replace(akt_rau, akt_rau %not_in% c(1, 2, 3), NA),
      # reorder the levels of alkrisk
      alkrisk = reorder_alkrisk_levels(alkrisk),
      # # retain the 10 most common cities and join the others into one category
      # stadt_bl = ifelse(
      #   stadt_bl %in% names(tail(sort(table(data$stadt_bl)), 10)),
      #   stadt_bl,
      #   "other"
      # ),
      # create discrete alter_ezig bins and make NA a category
      alter_ezig = addNA(cut_number(alter_ezig, n = 4))
    )

  # Make all categorical variables into factors
  categorical_vars <- vars_info %>% filter(categorical) %>% pull(name)
  data <- data %>% mutate(across(all_of(categorical_vars), as_factor))

  return(data)
}



#' Imputes systematic skips in dataset.
#'
#' Replaces allowed skips with negative results.
#'
#' @export
impute_dataset_systematic <- function(data) {
  # Check that the input data is a data frame
  assert_data_frame(data)

  # Replace allowed skip NAs
  data <- data %>%
    replace_na(list(Anzahl = 0)) # TODO: change once we have the tagged NAs
    # TODO: add further imputation of systematic missing values

  return(data)
}
