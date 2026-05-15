# Script for creating plots about the variables

renv::restore()

library(dplyr)

source(file.path("utils", "plotting_utils.R"))
source(file.path("utils", "saving_utils.R"))


# Import data
data_full <- load_data_bundle(file.path("data/preprocessed/data_full"))
data_discov <- load_data_bundle(file.path("data/preprocessed/data_discov_full"))
data_valid <- load_data_bundle(file.path("data/preprocessed/data_valid_full"))


# Set outputs folder
outputs_folder <- "outputs/EDA_plots"


# Plot number of variables per group
vars_per_group_plot <- data_full$meta |>
  count(group) |>
  ggplot(aes(x = reorder(group, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(title = "Variables per Group", x = "", y = "Count") +
  theme_ggplot
ggsave(
  filename = file.path(outputs_folder, "variables_per_group.png"),
  plot = vars_per_group_plot,
  width = 6,
  height = 4,
  dpi = dpi_ggplot
)


# Demographic Structure of ESA Sample
# A) age distribution

age_plot <- ggplot(data_discov$data, aes(x = alter)) +
  geom_histogram(bins = 25) +
  labs(title = "Age Distribution", x = "Age", y = "Count") +
  theme_ggplot
ggsave(
  filename = file.path(outputs_folder, "age_plot.png"),
  plot = age_plot,
  width = 3,
  height = 4,
  dpi = dpi_ggplot
)


# B) weighted age distribution
df <- data_discov$data |> filter(!is.na(alter), !is.na(wichtn))

age_weighted_plot <- ggplot(df, aes(x = alter, weight = wichtn)) +
  geom_histogram(bins = 25) +
  labs(title = "Weighted Age Distribution", x = "Age", y = "Weighted Count") +
  theme_ggplot
ggsave(
  filename = file.path(outputs_folder, "age_weighted_plot.png"),
  plot = age_weighted_plot,
  width = 3,
  height = 4,
  dpi = dpi_ggplot
)

###### missingness before imputing skip NAs per group

###### missingness after imputing skip NAs per group
###### missingness after imputing skip NAs per group

# Helper: find variable-name column in meta
var_col <- intersect(c("variable", "var", "name", "column"), names(data_full$meta))[1]

missingness_after_skip_plot <- data_full$data |>
  summarise(across(everything(), ~ mean(is.na(.x)) * 100)) |>
  tidyr::pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "missing_pct"
  ) |>
  left_join(
    data_full$meta |>
      rename(variable = all_of(var_col)) |>
      select(variable, group),
    by = "variable"
  ) |>
  filter(!is.na(group)) |>
  group_by(group) |>
  summarise(
    mean_missing_pct = mean(missing_pct, na.rm = TRUE),
    .groups = "drop"
  ) |>
  ggplot(aes(x = reorder(group, mean_missing_pct), y = mean_missing_pct)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Missingness After Imputing Skip NAs per Group",
    x = "",
    y = "Mean Missingness (%)"
  ) +
  theme_ggplot

ggsave(
  filename = file.path(outputs_folder, "missingness_after_skip_NAs_per_group.png"),
  plot = missingness_after_skip_plot,
  width = 6,
  height = 4,
  dpi = dpi_ggplot
)











###### missingness after imputing skip NAs per group (absolute counts)

library(tidyr)

# Helper: identify variable column in metadata
var_col <- intersect(c("variable", "var", "name", "column"), names(data_full$meta))[1]

missingness_after_skip_plot <- data_full$data |>
  summarise(across(everything(), ~ sum(is.na(.x)))) |>
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "n_missing"
  ) |>
  left_join(
    data_full$meta |>
      rename(variable = all_of(var_col)) |>
      select(variable, group),
    by = "variable"
  ) |>
  filter(!is.na(group)) |>
  group_by(group) |>
  summarise(
    total_missing = sum(n_missing, na.rm = TRUE),
    .groups = "drop"
  ) |>
  ggplot(aes(x = reorder(group, total_missing), y = total_missing)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Remaining Missing Values After Skip-NA Imputation",
    x = "",
    y = "Number of Missing Values"
  ) +
  theme_ggplot

ggsave(
  filename = file.path(outputs_folder, "missingness_after_skip_NAs_per_group_absolute.png"),
  plot = missingness_after_skip_plot,
  width = 6,
  height = 4,
  dpi = dpi_ggplot
)





###### missingness per variable: one plot per group

library(tidyr)

var_col <- intersect(c("variable", "var", "name", "column"), names(data_full$meta))[1]

missingness_per_variable_group <- data_full$data |>
  summarise(across(everything(), ~ sum(is.na(.x)))) |>
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "n_missing"
  ) |>
  left_join(
    data_full$meta |>
      rename(variable = all_of(var_col)) |>
      select(variable, group),
    by = "variable"
  ) |>
  filter(!is.na(group))

for (g in unique(missingness_per_variable_group$group)) {

  df_group <- missingness_per_variable_group |>
    filter(group == g) |>
    arrange(n_missing)

  p <- ggplot(df_group, aes(x = reorder(variable, n_missing), y = n_missing)) +
    geom_col() +
    coord_flip() +
    labs(
      title = paste("Remaining Missing Values:", g),
      x = "",
      y = "Number of Missing Values"
    ) +
    theme_ggplot +
    theme(axis.text.y = element_text(size = 7))

  safe_group_name <- gsub("[^A-Za-z0-9_-]", "_", g)

  ggsave(
    filename = file.path(
      outputs_folder,
      paste0("missingness_after_skip_NAs_", safe_group_name, ".png")
    ),
    plot = p,
    width = 8,
    height = max(4, 0.25 * nrow(df_group)),
    dpi = dpi_ggplot
  )
}
