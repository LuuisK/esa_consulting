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
data_discov_reduced <- load_data_bundle(file.path("data", "preprocessed", "data_discov_reduced"))
data_valid_reduced <- load_data_bundle(file.path("data", "preprocessed", "data_valid_reduced"))

outputs_folder <- file.path("outputs", "EDA_plots")
dir.create(outputs_folder, recursive = TRUE, showWarnings = FALSE)

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

# ------------------------------------------------------------------------------
# 1. Variables per group
# ------------------------------------------------------------------------------

variables_per_group_plot <- bind_rows(
  data_discov_reduced$meta |>
    count(group) |>
    mutate(year = "2018"),
  
  data_valid_reduced$meta |>
    count(group) |>
    mutate(year = "2021")
) |>
  ggplot(aes(x = group, y = n, fill = year)) +
  geom_col(position = position_dodge(width = 0.9)) +
  coord_flip() +
  scale_fill_manual(values = colors_year) +
  labs(
    title = "Variables per Group in the Reduced Feature Spaces",
    x = "",
    y = "Number of variables",
    fill = "Year"
  ) +
  theme_ggplot

save_plot(
  variables_per_group_plot,
  "variables_per_group_reduced_feature_spaces.png",
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
# 4. Weighted gender distribution in 2018 and 2021
# ------------------------------------------------------------------------------

gender_data <- data_full$data |>
  filter(
    zeit %in% c(2018, 2021),
    !is.na(ges),
    !is.na(wichtn)
  ) |>
  mutate(
    year = factor(zeit),
    gender = case_when(
      as.character(ges) == "1" ~ "Male",
      as.character(ges) == "2" ~ "Female",
      TRUE ~ as.character(ges)
    )
  ) |>
  group_by(year, gender) |>
  summarise(
    n_unweighted = n(),
    n_weighted = sum(wichtn),
    .groups = "drop"
  )

gender_plot <- gender_data |>
  ggplot(
    aes(
      x = gender,
      y = n_weighted,
      fill = year
    )
  ) +
  geom_col(
    position = position_dodge(width = 0.9)
  ) +
  geom_text(
    aes(label = round(n_weighted, 1)),
    position = position_dodge(width = 0.9),
    vjust = -0.3,
    size = 3.5,
    family = font_name
  ) +
  scale_fill_manual(values = colors_year) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.12))
  ) +
  labs(
    title = "Weighted Gender Distribution in 2018 and 2021",
    x = "Gender",
    y = "Weighted Count",
    fill = "Year"
  ) +
  theme_ggplot

save_plot(
  gender_plot,
  "weighted_gender_distribution_2018_2021.png",
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
# 5. Weighted interview types in 2018 and 2021
# ------------------------------------------------------------------------------

interview_type_plot <- data_full$data |>
  filter(
    zeit %in% c(2018, 2021),
    !is.na(int_art),
    !is.na(wichtn)
  ) |>
  mutate(
    year = factor(zeit),
    interview_type = case_when(
      as.character(int_art) == "1" ~ "CATI",
      as.character(int_art) == "2" ~ "PAPI",
      as.character(int_art) == "3" ~ "CAWI",
      TRUE ~ as.character(int_art)
    )
  ) |>
  group_by(year, interview_type) |>
  summarise(
    weighted_n = sum(wichtn),
    .groups = "drop"
  ) |>
  ggplot(aes(x = interview_type, y = weighted_n, fill = year)) +
  geom_col(position = position_dodge(width = 0.9)) +
  geom_text(
    aes(label = round(weighted_n, 1)),
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
    y = "Weighted count",
    fill = "Year"
  ) +
  theme_ggplot

save_plot(
  interview_type_plot,
  "weighted_interview_types_2018_2021.png",
  width = 6,
  height = 4
)

# ------------------------------------------------------------------------------
# 6. Average missingness per variable group: discovery
# ------------------------------------------------------------------------------

var_col_discov <- get_var_col(data_discov_reduced$meta)

missingness_discovery_plot <- data_discov_reduced$data |>
  summarise(across(everything(), ~ mean(is.na(.x)) * 100)) |>
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "missing_pct"
  ) |>
  left_join(
    data_discov_reduced$meta |>
      rename(variable = all_of(var_col_discov)) |>
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
    title = "Average Missingness per Variable Group: Discovery",
    subtitle = "Mean percentage of missing values across retained clustering variables",
    x = "",
    y = "Mean missingness (%)"
  ) +
  theme_ggplot

save_plot(
  missingness_discovery_plot,
  "average_missingness_by_group_discovery.png",
  width = 7,
  height = 5
)

# ------------------------------------------------------------------------------
# 7. Average missingness per variable group: validation
# ------------------------------------------------------------------------------

var_col_valid <- get_var_col(data_valid_reduced$meta)

missingness_validation_plot <- data_valid_reduced$data |>
  summarise(across(everything(), ~ mean(is.na(.x)) * 100)) |>
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "missing_pct"
  ) |>
  left_join(
    data_valid_reduced$meta |>
      rename(variable = all_of(var_col_valid)) |>
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
    title = "Average Missingness per Variable Group: Validation",
    subtitle = "Mean percentage of missing values across retained clustering variables",
    x = "",
    y = "Mean missingness (%)"
  ) +
  theme_ggplot

save_plot(
  missingness_validation_plot,
  "average_missingness_by_group_validation.png",
  width = 7,
  height = 5
)
