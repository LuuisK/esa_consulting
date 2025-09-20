#' Script for preprocessing.
#'
#' Reads in the dataset and preprocesses it.



# Load required packages and functions from funcs.R
box::use(
  haven[read_dta],
  dplyr[`%>%`, filter, pull, select, all_of, mutate, bind_rows],
  funcs = ./funcs[`%not_in%`]
)



# # Set a random seed for reproducibility
# set.seed(9876) UNCOMMENT WHEN USING PROBABILISTIC IMPUTATION

# Hardcoded paths, change to apply to new data
data_folder <- file.path("data", "raw")
save_folder <- file.path("data", "preprocessed")
outputs_folder <- "outputs"
data_filename <- "ESA1821_neu_SO.dta"



# Read in data
data_path <- file.path(data_folder, data_filename)
esa_2018_2021 <- read_dta(data_path)



# Create data frame containing information about variables
var_labels <- vapply(
  esa_2018_2021, function(x) attr(x, "label"), FUN.VALUE = character(1)
)
vars_info <- data.frame(name = names(var_labels), label = unname(var_labels))
vars_info <- funcs$sort_var_groups(vars_info, esa_2018_2021)
vars_info <- funcs$sort_var_types(vars_info, esa_2018_2021)

# Remove NA variables from the dataset
all_na_vars <- vars_info %>% filter(all_na) %>% pull(name)
esa_2018_2021 <- esa_2018_2021 %>% select(- all_of(all_na_vars))
vars_info <- vars_info %>% filter(name %not_in% all_na_vars)

# Correct errors, reformat
esa_2018_2021 <- funcs$correct_and_reformat(esa_2018_2021, vars_info)



# Split into 2018 and 2021 datasets
esa_2018 <- esa_2018_2021 %>% filter(zeit == 2018)
esa_2021 <- esa_2018_2021 %>% filter(zeit == 2021)



# Replace systematic skip values
esa_2018 <- funcs$impute_dataset_systematic(esa_2018)
esa_2021 <- funcs$impute_dataset_systematic(esa_2021)
esa_2018_2021 <- bind_rows(esa_2018, esa_2021)



# Save preprocessed datasets
funcs$save_data(
  esa_2018_2021,
  file.path(save_folder, "esa_2018_2021.rds"),
  label = "preprocessed full dataset"
)
funcs$save_data(
  esa_2018,
  file.path(save_folder, "esa_2018.rds"),
  label = "preprocessed 2018 dataset"
)
funcs$save_data(
  esa_2021,
  file.path(save_folder, "esa_2021.rds"),
  label = "preprocessed 2021 dataset"
)
funcs$save_data(
  vars_info,
  file.path(save_folder, "vars_info.rds"),
  label = "information about the variables"
)
