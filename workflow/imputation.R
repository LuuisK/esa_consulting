#' Notebook for imputation
#' Run notebooks in the following order: preprocessing.ipynb -> imputation.R ->
#' dimensionality_reduction.R -> gmm_clustering.ipynb

# Load required packages
library(checkmate) # assertions
library(dplyr) # data-frame methods
library(mice) # mice imputation

source(file.path("utils/saving_utils.R"))


# Read in data
data_discov <- load_data_bundle(
    file.path("data/preprocessed/data_discov_reduced")
)
data_valid <- load_data_bundle(
    file.path("data/preprocessed/data_valid_reduced")
)


# Set outputs folder
outputs_folder <- "data/imputed"

# Set number of MICE iterations
mice_iterations <- 5


#' Function for MICE
do_mice <- function(data, seed = NA, ...) {
    assert_data_frame(data)

    data |>
        mutate(across(where(is.factor), droplevels)) |>
        mice::mice(
            ...,
            #' Use polyreg instead of equivalent logreg for binary data because
            #' of convergence issues due to the maxit argument of logreg,
            #' see: https://stackoverflow.com/questions/58831710/
            #' mice-glm-fit-algorithm-did-not-converge/60056364#60056364
            defaultMethod = c("pmm", "polyreg", "polyreg", "polr"),
            # #' Not removing collinearity can destroy some correlation,
            # #' see: https://stefvanbuuren.name/fimd/sec-modelform.html
            # remove.collinear = FALSE,
            nnet.MaxNWts = 10000,
            # ridge=0.001,
            seed = seed
        )
}

# Train imputation kernels for discovery and validation data
mice_discov <- do_mice(
    data_discov$data,
    m = 1,
    maxit = mice_iterations,
    seed = 3523
)
mice_valid <- do_mice(
    data_valid$data,
    m = 1,
    maxit = mice_iterations,
    seed = 4363
)

# Impute data using the fitted kernels
data_discov_imputed <- list(
    data = mice::complete(mice_discov, action = 1),
    meta = data_discov$meta,
    na = data_discov$na
)
data_valid_imputed <- list(
    data = mice::complete(mice_valid, action = 1),
    meta = data_valid$meta,
    na = data_valid$na
)

# Save models
save_object(
    mice_discov,
    file.path(outputs_folder, "imputation_kernel_discov.rds")
)
save_object(
    mice_valid,
    file.path(outputs_folder, "imputation_kernel_valid.rds")
)

# Save imputed data
save_data_bundle(
    data_discov_imputed,
    file.path(outputs_folder, "data_discov_imputed")
)
save_data_bundle(
    data_valid_imputed,
    file.path(outputs_folder, "data_valid_imputed")
)
