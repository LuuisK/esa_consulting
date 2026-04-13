# Notebook for dimensionality reduction
#' Run notebooks in the following order: preprocessing.ipynb -> imputation.R ->
#' dimensionality_reduction.R -> gmm_clustering.ipynb

# Load required packages
library(checkmate) # assertions
library(dplyr) # data-frame methods
library(FactoMineR) # MFA

source(file.path("utils/saving_utils.R"))


# Read in data
data_discov <- load_data_bundle(file.path("data/imputed/data_discov_imputed"))
data_valid <- load_data_bundle(file.path("data/imputed/data_valid_imputed"))

# Set outputs folder
outputs_folder <- "outputs/dim_red"

# Set number of MFA components
n_comp <- 10


#' Function for MFA
do_mfa <- function(data, n_comp) {
    assert_list(data)
    assert_int(n_comp, lower = 0)

    vars_sorted <- data$meta |> arrange(group) |> pull(name)
    data_sorted <- data$data[vars_sorted]
    type_fn <- function(type) {
        if (all(type == "continuous")) {
            "s"
        } else if (all(type == "nominal" | type == "ordinal")) {
            "n"
        } else {
            "m"
        }
    }
    group_info <- data$meta |>
        group_by(group) |>
        summarize(size = n(), type = type_fn(type))

    MFA(
        data_sorted,
        group = group_info$size,
        type = group_info$type,
        ncp = n_comp
    )
}

# Fit models
mfa_discov <- do_mfa(data_discov, n_comp = n_comp)
mfa_valid <- do_mfa(data_valid, n_comp = n_comp)

# Apply models
data_discov_mfa <- tibble(mfa_discov$ind$coord)
data_valid_mfa <- tibble(mfa_discov$ind$coord)
vars_sorted <- data_valid$meta |> arrange(group) |> pull(name)
data_tf_mfa <- mfa_discov |> predict(data_valid$data[vars_sorted])


# Create plots ...
...


# Save models
save_object(mfa_discov, save_path = file.path(outputs_folder, "mfa_discov.rds"))
save_object(mfa_valid, save_path = file.path(outputs_folder, "mfa_valid.rds"))

# Save transformed data
save_data_bundle(data_discov_mfa, file.path(outputs_folder, "data_discov_mfa"))
save_data_bundle(data_valid_mfa, file.path(outputs_folder, "data_valid_mfa"))
save_data_bundle(data_tf_mfa, file.path(outputs_folder, "data_tf_mfa"))
