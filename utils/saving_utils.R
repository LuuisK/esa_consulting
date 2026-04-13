#' Module with utilities for saving.

# Load required packages
library(arrow) # read and write parquet
library(checkmate) # assertions
library(readr) # better rds functions


#' Save data bundle
#'
#' A data bundle is a list with a `.$data` field containing a dataset and
#' optionally a `.$meta` field and an `.$na` field containing metadata about
#' variables and missingness information respectively.
#'
#' The path may not have an extension. If they exist, metadata about variables
#' and missingness info are saved as well.
save_data_bundle <- function(data, path) {
    assert_list(data)
    assert_string(path)

    if (tools::file_ext(path) != "") {
        stop("'path' may not have an extension.")
    }

    if (is.null(data$data)) {
        stop("'data$data' may not be NULL.")
    }
    write_parquet(data$data, paste0(path, ".parquet"))

    if (!is.null(data$meta)) {
        write_parquet(data$meta, paste0(path, "__meta.parquet"))
    }

    if (!is.null(data$na)) {
        write_parquet(data$na, paste0(path, "__na.parquet"))
    }

    sprintf("Saved dataset at %s.", path) |> print()
}


#' Read parquet and remove index columns
read_data <- function(path) {
    assert_string(path)

    data = read_parquet(path)
    data |> select(-matches("^__index_level_\\d+__$"))
}


#' Load data bundle
#'
#' A data bundle is a list with a `.$data` field containing a dataset and
#' optionally a `.$meta` field and an `.$na` field containing metadata about
#' variables and missingness information respectively.
#'
#' The path may not have an extension. If they exist, metadata about variables
#' and missingness info are loaded as well.
load_data_bundle <- function(path) {
    assert_string(path)

    if (tools::file_ext(path) != "") {
        stop("'path' may not have an extension.")
    }

    bundle <- list()

    data_path <- paste0(path, ".parquet")
    if (!file.exists(data_path)) {
        sprintf("Dataset not found: %s", data_path) |> stop()
    }
    bundle$data <- read_data(data_path)

    meta_path <- paste0(path, "__meta.parquet")
    bundle$meta <- if (file.exists(meta_path)) read_data(meta_path) else NULL

    na_path <- paste0(path, "__na.parquet")
    bundle$na <- if (file.exists(na_path)) read_data(na_path) else NULL

    bundle
}


#' Save an R object as an rds-file.
#'
#' Saves the passed data with a given file name and prints a message.
save_object <- function(object, save_path) {
    assert_string(save_path)

    # Save as an rds-file and print message
    ext <- tools::file_ext(save_path)
    if (ext != "rds") {
        stop("The path must have an rds extension.")
    }
    write_rds(object, file = save_path)
    sprintf(
        "Saved %s as an %s file to %s.",
        deparse(substitute(object)),
        ext,
        save_path
    ) |>
        print()
}


#' Load an R object from an rds-file.
load_object <- function(save_path) {
    assert_string(save_path)

    # Read rds-file
    ext <- tools::file_ext(save_path)
    if (ext != "rds") {
        stop("The path must have an rds extension.")
    }

    read_rds(file = save_path)
}
