# Star plots of the cluster profiles
# Weighted shares (wichtn) among valid responses, scale 0-100 %.

library(arrow)
library(dplyr)
library(fmsb)
source("utils/saving_utils.R")

discov <- load_data_bundle("data/preprocessed/data_discov_full")
valid  <- load_data_bundle("data/preprocessed/data_valid_full")
lab_D  <- read_parquet("outputs/gmm_clustering/cluster_pred_discov.parquet")$label
lab_V  <- read_parquet("outputs/gmm_clustering/cluster_pred_valid.parquet")$label

# Axes counter-clockwise from the top (name = label shown in the plot)
AXES <- c(
  "No alcohol" = "No\nalcohol", "Low-risk alcohol" = "Low-risk\nalcohol",
  "Risky alcohol" = "Risky\nalcohol", "Smoker" = "Smoker", "Ex-smoker" = "Ex-\nsmoker",
  "Never smoked" = "Never\nsmoked", "Painkillers" = "Pain-\nkillers",
  "Psychotropic med." = "Psycho-\ntropic", "Rare med." = "Rare\nmed.",
  "Cannabis" = "Cannabis", "Other drugs" = "Other\ndrugs"
)

num <- function(x) as.numeric(as.character(x))

# 1 if at least one variable differs from `none`; 0 if all are validly `none`; otherwise NA
any_use <- function(d, vars, none) {
  m <- sapply(d[vars], function(x) num(x) != none)
  ifelse(rowSums(m, na.rm = TRUE) > 0, 1, ifelse(rowSums(is.na(m)) == 0, 0, NA))
}

# Codes: alk30kat 0 = none, 1 = low-risk, 2-4 = risky to high consumption;
# typ2 1 = non-smoker, 2 = ex-smoker, 3 = smoker;
# *_kat 0 = no use (past 30 days); fr12_* 1 = no use (past 12 months).
# fr12_meth1 is excluded because it was not asked in 2018.
indicators <- function(d) {
  alk <- num(d$alk30kat)
  typ <- num(d$typ2)
  tibble(
    "No alcohol"        = alk == 0,
    "Low-risk alcohol"  = alk == 1,
    "Risky alcohol"     = alk >= 2,
    "Smoker"            = typ == 3,
    "Ex-smoker"         = typ == 2,
    "Never smoked"      = typ == 1,
    "Painkillers"       = any_use(d, "schm_kat", 0),
    "Psychotropic med." = any_use(d, c("schl_kat", "ber_kat", "depr_kat", "neur_kat"), 0),
    "Rare med."         = any_use(d, c("anr_kat", "app_kat", "anab_kat"), 0),
    "Cannabis"          = any_use(d, "fr12_ha1", 1),
    "Other drugs"       = any_use(d, c("fr12_ec1", "fr12_ls1", "fr12_he1", "fr12_ao1", "fr12_ko1",
                                       "fr12_cr1", "fr12_pi1", "fr12_nps1", "fr12_au1"), 1)
  ) |> mutate(across(everything(), as.numeric))
}

# One row per cluster: unweighted n, weighted n and weighted share per axis
profile <- function(bundle, labels, prefix) {
  ind <- indicators(bundle$data) |> mutate(w = num(bundle$data$wichtn))
  bind_rows(mutate(ind, group = paste0(prefix, labels)), mutate(ind, group = "Total")) |>
    group_by(group) |>
    summarise(n = n(), n_w = sum(w),
              across(all_of(names(AXES)), \(x) weighted.mean(x, w, na.rm = TRUE))) |>
    arrange(suppressWarnings(as.numeric(substring(group, 2))))  # Total comes last
}

prof_D <- profile(discov, lab_D, "D")
prof_V <- profile(valid, lab_V, "V")


# One star per cluster in a 3 x 4 grid; grey = fewer than 100 weighted respondents
plot_wave <- function(prof, colour, file) {
  png(file, width = 13.33, height = 7.5, units = "in", res = 300)
  par(mfrow = c(3, 4), mar = c(0.5, 0.5, 2, 0.5))
  for (i in seq_len(nrow(prof))) {
    col <- if (prof$n_w[i] < 100) "grey50" else colour
    vals <- as.matrix(prof[i, names(AXES)])
    radarchart(as.data.frame(rbind(1, 0, vals)),  # rows: max, min, values
               centerzero = TRUE, axistype = 0, seg = 4,
               pcol = col, pfcol = adjustcolor(col, 0.35), plwd = 2,
               cglcol = "grey85", cglty = 1, vlabels = unname(AXES), vlcex = 0.8,
               title = sprintf("%s (n = %s)", prof$group[i], format(prof$n[i], big.mark = ",")))
  }
  dev.off()
}

dir.create("outputs/clustering", showWarnings = FALSE, recursive = TRUE)
plot_wave(prof_D, "#127A2E", "outputs/clustering/starplots_discov.png")
plot_wave(prof_V, "#151069", "outputs/clustering/starplots_valid.png")
