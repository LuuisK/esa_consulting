# Notebook for dimensionality reduction
#' Run notebooks in the following order: preprocessing.ipynb -> imputation.R ->
#' dimensionality_reduction.R -> gmm_clustering.ipynb

# Load required packages
library(arrow) # read and write parquet
library(checkmate) # assertions
library(dplyr) # data-frame methods
library(FactoMineR) # MFA

source(file.path("utils/saving_utils.R"))


# Read in data
data_discov <- load_data_bundle(file.path("data/imputed/data_discov_imputed"))
data_valid <- load_data_bundle(file.path("data/imputed/data_valid_imputed"))

# Set outputs folder
outputs_folder <- "data/dim_red"

# Set number of MFA components
n_comp <- 10


#' Function for MFA
do_mfa <- function(data, n_comp, ...) {
    assert_list(data)
    assert_int(n_comp, lower = 0)

    vars_sorted <- data$meta |> arrange(group) |> pull(name)
    data_sorted <- data$data[vars_sorted]
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
        ...
    )
}

# Fit models
mfa_discov <- do_mfa(data_discov, n_comp = n_comp, graph = F)
mfa_valid <- do_mfa(data_valid, n_comp = n_comp, graph = F)

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





# Eigenvalues / explained variance
mfa_discov$eig
mfa_valid$eig

# Contribution of variable groups to dimensions
mfa_discov$group$contrib
mfa_valid$group$contrib

install.packages("factoextra")
library(factoextra)
fviz_mfa_var(mfa_discov)
fviz_mfa_var(mfa_valid)


# # Coordinates of variable groups
# mfa_discov$group$coord
# mfa_valid$group$coord
#
# # Contribution of individual variables/categories
# mfa_discov$quanti.var$contrib
# mfa_discov$quali.var$contrib
#
# mfa_valid$quanti.var$contrib
# mfa_valid$quali.var$contrib
#
# data_discov$meta |> count(group, type)
# data_valid$meta |> count(group, type)
#
#
# dimdesc(mfa_discov, axes = 1:5)
# mfa_discov$quanti.var$coord
# mfa_discov$quali.var$eta2
#
#
#
# one_level_vars <- names(Filter(function(x) {
#     is.factor(x) && nlevels(droplevels(x)) < 2
# }, mfa_valid$call$X))
#
# one_level_vars
#
#
# mfa_valid_dimdesc <- mfa_valid
# mfa_valid_dimdesc$call$X <- mfa_valid_dimdesc$call$X[, !names(mfa_valid_dimdesc$call$X) %in% "fr12_au1"]
#
# dimdesc(mfa_valid_dimdesc, axes = 1:4)


