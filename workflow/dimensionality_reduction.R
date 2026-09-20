# Notebook for dimensionality reduction
# Run notebooks in the following order: preprocessing.ipynb -> imputation.R ->
# dimensionality_reduction.R -> gmm_clustering.ipynb

# Load required packages
library(arrow) # read and write parquet
library(checkmate) # assertions
library(dplyr) # data-frame methods
library(FactoMineR) # MFA
library(rlang) # data-masking

source(file.path("utils/saving_utils.R"))


# Read in data
data_discov <- load_data_bundle(file.path("data/imputed/data_discov_imputed"))
data_valid <- load_data_bundle(file.path("data/imputed/data_valid_imputed"))

# Set outputs folder
outputs_folder <- "data/dim_red"

# Set number of MFA components
n_comp <- 10


# Function for MFA
do_mfa <- function(data, n_comp, weight_var = NULL, ...) {
    assert_list(data)
    assert_int(n_comp, lower = 0)

    weight_var <- enquo(weight_var)

    weights <- NULL
    if (!quo_is_null(weight_var)) {
        weights <- data$data |> pull(!!weight_var)
        data <- list(
            data = data$data |> select(-!!weight_var),
            meta = data$meta |> filter(name != as_name(weight_var))
        )
    }

    vars_sorted <- data$meta |> arrange(group) |> pull(name)
    data_sorted <- data$data |> select(all_of(vars_sorted))
    type_fn <- function(type) {
        if (all(type == "continuous")) {
            "s"
        } else if (all(type %in% c("nominal", "ordinal", "binary"))) {
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
        ncp = n_comp,
        row.w = weights,
        ...
    )
}

# Fit models
mfa_discov <- do_mfa(data_discov, n_comp, weight_var = wichtn, graph = F)
mfa_valid <- do_mfa(data_valid, n_comp, weight_var = wichtn, graph = F)

# Apply models
data_discov_mfa <- as_tibble(mfa_discov$ind$coord)
data_valid_mfa <- as_tibble(mfa_valid$ind$coord)


# Save models
save_object(mfa_discov, save_path = file.path(outputs_folder, "mfa_discov.rds"))
save_object(mfa_valid, save_path = file.path(outputs_folder, "mfa_valid.rds"))

# Save transformed data
write_parquet(
    data_discov_mfa,
    file.path(outputs_folder, "data_discov_mfa.parquet")
)
write_parquet(
    data_valid_mfa,
    file.path(outputs_folder, "data_valid_mfa.parquet")
)




##############################################################
# Check Variance explained
##############################################################

component_to_check <- 15

# Discovery

explained_var_discov <- as_tibble(mfa_discov$eig, rownames = "component") |>
    rename(cum_per_var = `cumulative percentage of variance`) |>
    mutate(component = as.integer(sub("comp ", "", component)))

result_at_15_discov <- explained_var_discov |>
    filter(component == component_to_check)

cat(sprintf(
    "Discovery, component %d: %.2f%% cumulative variance\n",
    result_at_15_discov$component,
    result_at_15_discov$cum_per_var
))


# Validation
explained_var_valid <- as_tibble(mfa_valid$eig, rownames = "component") |>
    rename(cum_per_var = `cumulative percentage of variance`) |>
    mutate(component = as.integer(sub("comp ", "", component)))

result_at_15_valid <- explained_var_valid |>
    filter(component == component_to_check)

cat(sprintf(
    "Validation, component %d: %.2f%% cumulative variance\n",
    result_at_15_valid$component,
    result_at_15_valid$cum_per_var
))
