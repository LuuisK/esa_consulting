# Script for generating alternative imputations and MFA coordinates

# Load required packages
suppressPackageStartupMessages({
    library(arrow)
    library(checkmate)
    library(dplyr)
    library(FactoMineR)
    library(mice)
})

source(file.path("utils/saving_utils.R"))
source(file.path("utils/mfa_utils.R"))


#' Parse command line args n_imputations, seed, dimensions, imputation_dir, output_dir
args <- commandArgs(trailingOnly = TRUE)
assert_character(args, len = 6L, any.missing = FALSE, min.chars = 1L)
n_imputations <- as.numeric(args[[1]])
n_imputations <- assert_int(n_imputations, lower = 1L, coerce = TRUE)
seed <- as.numeric(args[[2]])
seed <- assert_int(
    seed,
    lower = 1L,
    upper = .Machine$integer.max - 1L,
    coerce = TRUE
)
dimensions <- c(discov = as.numeric(args[[3]]), valid = as.numeric(args[[4]]))
dimensions <- assert_integerish(
    dimensions,
    lower = 1L,
    len = 2L,
    any.missing = FALSE,
    coerce = TRUE
)
imputation_dir <- args[[5]]
assert_directory_exists(imputation_dir, access = "r")
output_dir <- args[[6]]
if (file.exists(output_dir)) {
    assert_directory_exists(output_dir, access = "w")
}


# Load the saved imputation specifications and variable metadata
reference_configs <- lapply(c("discov", "valid"), function(wave) {
    kernel_path <- file.path(
        imputation_dir,
        paste0("imputation_kernel_", wave, ".rds")
    )
    kernel <- load_object(kernel_path)
    if (!inherits(kernel, "mids")) {
        stop(kernel_path, " must be a mids object.")
    }
    meta <- read_data(file.path(
        imputation_dir,
        paste0("data_", wave, "_imputed__meta.parquet")
    ))

    list(
        kernel = kernel,
        meta = meta,
        weights = mice::complete(kernel, action = 1L)[["wichtn"]]
    )
})
names(reference_configs) <- c("discov", "valid")


# Run imputation and MFA
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
RNGkind(
    kind = "Mersenne-Twister",
    normal.kind = "Inversion",
    sample.kind = "Rejection"
)
for (wave in names(reference_configs)) {
    config <- reference_configs[[wave]]
    kernel <- config$kernel
    wave_seed <- seed + (match(wave, names(reference_configs)) - 1L)
    cat("Imputing ", wave, " with ", n_imputations, " new chains.\n", sep = "")

    # Generate new chains from data with missing values and imputation settings
    mice_alternative <- mice::mice(
        kernel$data,
        m = n_imputations,
        maxit = kernel$iteration,
        method = kernel$method,
        predictorMatrix = kernel$predictorMatrix,
        where = kernel$where,
        blocks = kernel$blocks,
        visitSequence = kernel$visitSequence,
        blots = kernel$blots,
        post = kernel$post,
        ignore = kernel$ignore,
        nnet.MaxNWts = 10000,
        seed = wave_seed,
        printFlag = FALSE
    )
    save_object(
        mice_alternative,
        file.path(output_dir, paste0("mice_alternative_", wave, ".rds"))
    )
    if (!is.null(mice_alternative$loggedEvents)) {
        write_parquet(
            mice_alternative$loggedEvents,
            file.path(
                output_dir,
                paste0("alternative_logged_events_", wave, ".parquet")
            )
        )
    }

    # Fit MFA on each completed wave and preserve respondent row order
    for (imputation in seq_len(n_imputations)) {
        alternative_imputed <- mice::complete(
            mice_alternative,
            imputation
        )
        alternative_scores <- do_mfa(
            list(data = alternative_imputed, meta = config$meta),
            dimensions[[wave]],
            weight_var = wichtn,
            graph = F
        )$ind$coord
        result <- data.frame(
            sample_weight = config$weights,
            as_tibble(alternative_scores),
            check.names = FALSE
        )
        path <- file.path(
            output_dir,
            sprintf("mfa_%s_%03d.parquet", wave, imputation)
        )
        write_parquet(result, path)
        cat("Saved ", path, "\n", sep = "")
    }
}
