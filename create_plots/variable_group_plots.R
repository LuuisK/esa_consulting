# EDA plots for ESA project

library(dplyr)
library(tidyr)
library(ggplot2)
library(showtext)
library(sysfonts)

source(file.path("utils", "plotting_utils.R"))
source(file.path("utils", "saving_utils.R"))

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------

data_full <- load_data_bundle(file.path("data", "preprocessed", "data_full"))
data_discov <- load_data_bundle(file.path("data", "preprocessed", "data_discov_full"))
data_valid <- load_data_bundle(file.path("data", "preprocessed", "data_valid_full"))

outputs_folder <- file.path("outputs", "EDA_plots")
dir.create(outputs_folder, recursive = TRUE)

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

get_var_col <- function(meta) {
  var_col <- intersect(c("variable", "var", "name", "column"), names(meta))[1]

  if (is.na(var_col)) {
    stop("No variable-name column found in metadata.")
  }

  var_col
}

save_plot <- function(plot, filename, width = 6, height = 4) {
  ggsave(
    filename = file.path(outputs_folder, filename),
    plot = plot,
    width = width,
    height = height,
    dpi = dpi_ggplot
  )
}

plot_average_missingness_by_group <- function(data_bundle, dataset_label) {
  var_col <- get_var_col(data_bundle$meta)

  data_bundle$data |>
    summarise(across(everything(), ~ mean(is.na(.x)) * 100)) |>
    pivot_longer(
      cols = everything(),
      names_to = "variable",
      values_to = "missing_pct"
    ) |>
    left_join(
      data_bundle$meta |>
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
    geom_col(fill = default_color) +
    coord_flip() +
    labs(
      title = paste("Average Missingness per Variable Group:", dataset_label),
      subtitle = "Mean percentage of missing values across variables in each group",
      x = "",
      y = "Mean missingness (%)"
    ) +
    theme_ggplot
}

# ------------------------------------------------------------------------------
# 1. Variables per group
# ------------------------------------------------------------------------------

variables_per_group_plot <- data_full$meta |>
  count(group) |>
  ggplot(aes(x = reorder(group, n), y = n)) +
  geom_col(fill = default_color) +
  coord_flip() +
  labs(
    title = "Variables per Group",
    x = "",
    y = "Number of variables"
  ) +
  theme_ggplot

save_plot(
  variables_per_group_plot,
  "variables_per_group.png",
  width = 6,
  height = 4
)

# ------------------------------------------------------------------------------
# 2. Age distribution
# ------------------------------------------------------------------------------

age_plot <- data_discov$data |>
  filter(!is.na(alter)) |>
  ggplot(aes(x = alter)) +
  geom_histogram(bins = 25, fill = default_color) +
  labs(
    title = "Age Distribution",
    x = "Age",
    y = "Count"
  ) +
  theme_ggplot

save_plot(
  age_plot,
  "age_distribution.png",
  width = 4,
  height = 4
)

# ------------------------------------------------------------------------------
# 3. Weighted age distribution
# ------------------------------------------------------------------------------

weighted_age_plot <- data_discov$data |>
  filter(!is.na(alter), !is.na(wichtn)) |>
  ggplot(aes(x = alter, weight = wichtn)) +
  geom_histogram(bins = 25, fill = default_color) +
  labs(
    title = "Weighted Age Distribution",
    x = "Age",
    y = "Weighted count"
  ) +
  theme_ggplot

save_plot(
  weighted_age_plot,
  "weighted_age_distribution.png",
  width = 4,
  height = 4
)

# ------------------------------------------------------------------------------
# 4. Gender distribution in 2018 and 2021
# ------------------------------------------------------------------------------

gender_plot <- data_full$data |>
  filter(zeit %in% c(2018, 2021), !is.na(ges)) |>
  mutate(
    year = factor(zeit),
    gender = case_when(
      as.character(ges) == "1" ~ "Male",
      as.character(ges) == "2" ~ "Female",
      TRUE ~ as.character(ges)
    )
  ) |>
  count(year, gender) |>
  ggplot(aes(x = gender, y = n, fill = year)) +
  geom_col(position = position_dodge(width = 0.9)) +
  geom_text(
    aes(label = n),
    position = position_dodge(width = 0.9),
    vjust = -0.3,
    size = 3.5,
    family = font_name
  ) +
  scale_fill_manual(values = colors_year) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "Gender Distribution in 2018 and 2021",
    x = "Gender",
    y = "Count",
    fill = "Year"
  ) +
  theme_ggplot

save_plot(
  gender_plot,
  "gender_distribution_2018_2021.png",
  width = 6,
  height = 4
)

# ------------------------------------------------------------------------------
# 5. Interview types in 2018 and 2021
# ------------------------------------------------------------------------------

interview_type_plot <- data_full$data |>
  filter(zeit %in% c(2018, 2021), !is.na(int_art)) |>
  mutate(
    year = factor(zeit),
    interview_type = case_when(
      as.character(int_art) == "1" ~ "CATI",
      as.character(int_art) == "2" ~ "PAPI",
      as.character(int_art) == "3" ~ "CAWI",
      TRUE ~ as.character(int_art)
    )
  ) |>
  count(year, interview_type) |>
  ggplot(aes(x = interview_type, y = n, fill = year)) +
  geom_col(position = position_dodge(width = 0.9)) +
  geom_text(
    aes(label = n),
    position = position_dodge(width = 0.9),
    vjust = -0.3,
    size = 3.5,
    family = font_name
  ) +
  scale_fill_manual(values = colors_year) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "Interview Types in 2018 and 2021",
    x = "Interview type",
    y = "Count",
    fill = "Year"
  ) +
  theme_ggplot

save_plot(
  interview_type_plot,
  "interview_types_2018_2021.png",
  width = 6,
  height = 4
)

# ------------------------------------------------------------------------------
# 6. Average missingness per variable group: discovery
# ------------------------------------------------------------------------------

missingness_discovery_plot <- plot_average_missingness_by_group(
  data_bundle = data_discov,
  dataset_label = "Discovery"
)

save_plot(
  missingness_discovery_plot,
  "average_missingness_by_group_discovery.png",
  width = 7,
  height = 5
)

# ------------------------------------------------------------------------------
# 7. Average missingness per variable group: validation
# ------------------------------------------------------------------------------

missingness_validation_plot <- plot_average_missingness_by_group(
  data_bundle = data_valid,
  dataset_label = "Validation"
)

save_plot(
  missingness_validation_plot,
  "average_missingness_by_group_validation.png",
  width = 7,
  height = 5
)
