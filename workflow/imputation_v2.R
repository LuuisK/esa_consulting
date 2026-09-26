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

# Fix make.method bug
data_discov$data <- as.data.frame(data_discov$data)
data_valid$data <- as.data.frame(data_valid$data)



# Leere Faktorlevels entfernen, bevor quickpred, make.method und
# problem_vars berechnet werden (sonst zählt table() leere Levels als 0)
data_discov$data <- droplevels(data_discov$data)
data_valid$data  <- droplevels(data_valid$data)


# ============================================================
# Discovery
# ============================================================

# Volle Standardmatrix, wie mice sie ohne quickpred verwenden würde
pred_full_discov <- make.predictorMatrix(data_discov$data)

# Gesamtzahl der Prädiktor-Beziehungen (Zellen, exkl. Diagonale)
n_total_discov <- sum(pred_full_discov == 1)

# Predictor Matrix via quickpred erzeugen, um Multikollinearität zu reduzieren
pred_discov <- quickpred(
  data_discov$data,
  mincor = 0.1
)

# Wie viele davon hat quickpred rausgeworfen?
n_dropped_discov <- sum(pred_full_discov == 1 & pred_discov == 0)

cat(sprintf("discov: %d von %d Prädiktor-Beziehungen entfernt (%.1f %%)\n",
            n_dropped_discov, n_total_discov,
            100 * n_dropped_discov / n_total_discov))

# ============================================================
# Validation
# ============================================================

# Volle Standardmatrix für Validation, separat von der discov-Matrix
pred_full_valid <- make.predictorMatrix(data_valid$data)

# Gesamtzahl der Prädiktor-Beziehungen (Zellen, exkl. Diagonale)
n_total_valid <- sum(pred_full_valid == 1)

pred_valid <- quickpred(
  data_valid$data,
  mincor = 0.1
)

# Wie viele davon hat quickpred rausgeworfen?
n_dropped_valid <- sum(pred_full_valid == 1 & pred_valid == 0)

cat(sprintf("valid: %d von %d Prädiktor-Beziehungen entfernt (%.1f %%)\n",
            n_dropped_valid, n_total_valid,
            100 * n_dropped_valid / n_total_valid))


min_category_count <- 10

# Discovery data
factor_vars_discov <- data_discov$data |>
  select(where(is.factor)) |>
  names()

# problem_vars speichert alle kategorialen Variablen, bei denen
# mindestens eine Kategorie weniger als 10 Beobachtungen hat.
problem_vars_discov <- factor_vars_discov[
  sapply(factor_vars_discov, function(v) {
    any(table(data_discov$data[[v]]) < min_category_count)
  })
]

meth_discov <- make.method(
  data_discov$data,
  defaultMethod = c("pmm", "polyreg", "polyreg", "polr")
)
# Gezielt cart für Variablen mit sehr seltenen Kategorien
meth_discov[problem_vars_discov] <- "cart"

# Validation data
factor_vars_valid <- data_valid$data |>
  select(where(is.factor)) |>
  names()


problem_vars_valid <- factor_vars_valid[
  sapply(factor_vars_valid, function(v) {
    any(table(data_valid$data[[v]]) < min_category_count)
  })
]

meth_valid <- make.method(
  data_valid$data,
  defaultMethod = c("pmm", "polyreg", "polyreg", "polr")
)
meth_valid[problem_vars_valid] <- "cart"

# Zur Kontrolle, welche Variablen auf cart umgestellt wurden:
# problem_vars_discov
# problem_vars_valid


# Set outputs folder
outputs_folder <- "data/imputed"

# Set number of MICE iterations
mice_iterations <- 40


# Function for MICE
do_mice <- function(data, seed = NA, ...) {
  assert_data_frame(data)


  if (!is.na(seed)){
    RNGkind(kind = "Mersenne-Twister",
            normal.kind = "Inversion",
            sample.kind = "Rejection"
    )
    set.seed(seed)
  }

  data |>
    mutate(across(where(is.factor), droplevels)) |>
    mice::mice(
      ...,
      #' Use polyreg instead of equivalent logreg for binary data because
      #' of convergence issues due to the maxit argument of logreg,
      #' see: https://stackoverflow.com/questions/58831710/
      #' mice-glm-fit-algorithm-did-not-converge/60056364#60056364
      # defaultMethod wird jetzt NICHT mehr genutzt, method kommt von außen (siehe method-Argument)
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
  predictorMatrix = pred_discov,
  method = meth_discov,
  seed = 3523
)
mice_valid <- do_mice(
  data_valid$data,
  m = 1,
  maxit = mice_iterations,
  predictorMatrix = pred_valid,
  method = meth_valid,
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


data_before <- load_data_bundle(
  "data/preprocessed/data_discov_reduced"
)

data_after <- load_data_bundle(
  "data/imputed/data_discov_imputed"
)

before_df <- as.data.frame(data_before$data)
after_df  <- as.data.frame(data_after$data)

na_pos <- which(is.na(before_df), arr.ind = TRUE)

data.frame(
  row = na_pos[1:10, 1],
  variable = names(before_df)[na_pos[1:10, 2]],
  imputed_value = sapply(
    1:10,
    function(i) after_df[
      na_pos[i, 1],
      na_pos[i, 2]
    ]
  )
)

#
# # ============================================================
# # MICE convergence diagnostics
# # ============================================================
#
# # ---- Discovery data ----
#
# imp_diag_discov <- do_mice(
#   data_discov$data,
#   m = 5,
#   maxit = 40,
#   predictorMatrix = pred_discov,
#   method = meth_discov,
#   seed = 3523
# )
#
# # Show number of iterations
# imp_diag_discov$iteration
#
# # Trace plots
# plot(imp_diag_discov)
#
# # Numerical convergence diagnostics
# conv_discov <- mice::convergence(imp_diag_discov)
#
# print(conv_discov)
#
#
# # ---- Validation data ----
#
# imp_diag_valid <- do_mice(
#   data_valid$data,
#   m = 5,
#   maxit = 40,
#   predictorMatrix = pred_valid,
#   method = meth_valid,
#   seed = 4363
# )
#
# # Show number of iterations
# imp_diag_valid$iteration
#
# # Trace plots
# plot(imp_diag_valid)
#
# # Numerical convergence diagnostics
# conv_valid <- mice::convergence(imp_diag_valid)
#
# print(conv_valid)
#
#
#
#
#
#
#
#
#
#
#
# # Häufigste Event-Typen (out-Feld) bei discov
# sort(
#   table(imp_diag_discov$loggedEvents$out),
#   decreasing = TRUE
# )[1:20]
#
# # Welche Variablen (dep) am häufigsten betroffen sind
# sort(
#   table(imp_diag_discov$loggedEvents$dep),
#   decreasing = TRUE
# )[1:20]
#
# # Welche Methode am häufigsten betroffen ist
# table(imp_diag_discov$loggedEvents$meth)
#
# # Kombiniert: Variable x Methode x Ereignistyp, häufigste zuerst
# imp_diag_discov$loggedEvents |>
#   dplyr::count(dep, meth, out, sort = TRUE) |>
#   head(30)
#
#
#
# table(data_discov$data$vx210)
# table(data_valid$data$typ2)
# table(data_valid$data$fag_sch)
# table(data_valid$data$Anzahl)
# table(data_discov$data$fr12_nps1)
#
#
# # Systematisch prüfen: welche Variablen haben am Ende (letzte Iteration) einen psrf über 1.2?
#
# last_it <- max(conv_discov$.it, na.rm = TRUE)
#
# problem_convergence_discov <- conv_discov |>
#   dplyr::filter(.it == last_it, !is.na(psrf), psrf > 1.2) |>
#   dplyr::arrange(dplyr::desc(psrf))
#
# print(problem_convergence_discov)
#
# # Gleiches für den Validation-Datensatz
# problem_convergence_valid <- conv_valid |>
#   dplyr::filter(.it == last_it, !is.na(psrf), psrf > 1.2) |>
#   dplyr::arrange(dplyr::desc(psrf))
#
# print(problem_convergence_valid)
