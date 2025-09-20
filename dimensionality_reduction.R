#' Dimensionality reduction script.
#'
#' Uses factor analysis to find latent dimensions. Creates graphs visualizing
#' the latent dimensions.



# Load required packages and functions from funcs.R
box::use(
  mice,
  missMDA[imputeFAMD],
  FactoMineR[FAMD],
  readr[read_rds],
  dplyr[`%>%`, filter, pull, select, all_of],
  dplyr[...],
  tibble[...],
  funcs = ./funcs[`%not_in%`]
)



# Hardcoded paths, change to apply to new data
data_folder <- file.path("data", "preprocessed")
outputs_folder <- "outputs"
data_filename <- "esa_2018_2021.rds"
vars_info_filename <- "vars_info.rds"



# Load dataset and variable information
data_path <- file.path(data_folder, data_filename)
esa_2018_2021 <- read_rds(data_path)

vars_info_path <- file.path(data_folder, vars_info_filename)
vars_info <- read_rds(vars_info_path)

# Separate out the alcohol variables
alcohol_vars <- vars_info %>% filter(group == "alcohol") %>% pull(name)
alcohol_data <- esa_2018_2021 %>% select(all_of(alcohol_vars))

# Impute completely random NAs in dataset
imputed_data <- imputeFAMD(
  alcohol_data, ncp = 2, method = "Regularized", coeff.ridge = 1
)



# Conduct factor analysis for mixed data
n_latent <- 10
alcohol_famd <- FAMD(
  alcohol_data,
  ncp = n_latent,
  graph = TRUE,
  tab.disj = imputed_data$tab.disj
)
plot(alcohol_famd)
