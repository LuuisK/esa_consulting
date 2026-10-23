#' Module with utilities for dimensionality reduction

# Load required packages
library(checkmate) # assertions
library(dplyr) # data-frame methods
library(FactoMineR) # MFA
library(rlang) # data-masking


# Function for MFA
do_mfa <- function(data, n_comp, weight_var = NULL, ...) {
    assert_list(data)
    assert_int(n_comp, lower = 1)

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
