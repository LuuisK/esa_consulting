# Script for creating plots about the variables

renv::restore()

library(dplyr)
library(tidyr)
library(ggplot2)
library(showtext)
library(sysfonts)

source(file.path("utils", "plotting_utils.R"))
source(file.path("utils", "saving_utils.R"))

font_add(
  family = "Libertinus",
  regular = file.path("fonts", "Libertinus-7.051", "LibertinusSerif-Regular.otf"),
  bold = file.path("fonts", "Libertinus-7.051", "LibertinusSerif-Bold.otf"),
  italic = file.path("fonts", "Libertinus-7.051", "LibertinusSerif-Italic.otf"),
  bolditalic = file.path("fonts", "Libertinus-7.051", "LibertinusSerif-BoldItalic.otf")
)

showtext_auto()

theme_ggplot <- theme_ggplot +
  theme(
    text = element_text(family = "Libertinus", size = 14),
    plot.title = element_text(family = "Libertinus", face = "bold", size = 18),
    plot.subtitle = element_text(family = "Libertinus", size = 14),
    axis.title = element_text(family = "Libertinus", size = 14),
    axis.text = element_text(family = "Libertinus", size = 12),
    legend.title = element_text(family = "Libertinus", size = 13),
    legend.text = element_text(family = "Libertinus", size = 12),
    strip.text = element_text(family = "Libertinus", size = 13)
  )

# Import data
data_full <- load_data_bundle(file.path("data/preprocessed/data_full"))
data_discov <- load_data_bundle(file.path("data/preprocessed/data_discov_full"))
data_valid <- load_data_bundle(file.path("data/preprocessed/data_valid_full"))

# Set outputs folder
outputs_folder <- file.path("outputs", "EDA_plots")

if (!dir.exists(outputs_folder)) {
  dir.create(outputs_folder, recursive = TRUE)
}

# Plot number of variables per group
vars_per_group_plot <- data_full$meta |>
  count(group) |>
  ggplot(aes(x = reorder(group, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(title = "Variables per Group", x = "", y = "Count") +
  theme_ggplot +
  theme(plot.margin = margin(t = 10, r = 120, b = 10, l = 10))


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

###### missingness after imputing skip NAs per group

var_col <- intersect(c("variable", "var", "name", "column"), names(data_full$meta))[1]

missingness_after_skip_plot <- data_full$data |>
  summarise(across(everything(), ~ mean(is.na(.x)) * 100)) |>
  pivot_longer(
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

missingness_after_skip_absolute_plot <- data_full$data |>
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
  plot = missingness_after_skip_absolute_plot,
  width = 6,
  height = 4,
  dpi = dpi_ggplot
)

###### missingness per variable: one plot per group

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

###### gender distribution in 2018 and 2021

gender_plot_data <- data_full$data |>
  filter(zeit %in% c(2018, 2021), !is.na(ges)) |>
  mutate(
    zeit = factor(zeit),
    ges = as.character(ges),
    gender = case_when(
      ges == "1" ~ "Male",
      ges == "2" ~ "Female",
      TRUE ~ ges
    )
  ) |>
  count(zeit, gender)

gender_plot <- ggplot(
  gender_plot_data,
  aes(x = gender, y = n, fill = zeit)
) +
  geom_col(position = position_dodge(width = 0.9)) +
  geom_text(
    aes(label = n),
    position = position_dodge(width = 0.9),
    vjust = -0.3,
    size = 3.5
  ) +
  labs(
    title = "Gender Distribution in 2018 and 2021",
    x = "Gender",
    y = "Count",
    fill = "Year"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  theme_ggplot

ggsave(
  filename = file.path(outputs_folder, "gender_distribution_2018_2021.png"),
  plot = gender_plot,
  width = 6,
  height = 4,
  dpi = dpi_ggplot
)

### number of interview types in 2018 and 2021

# Fieldwork was conducted between May and September 2021 by the infas Institute for
# Applied Social Sciences GmbH (infas Institut für angewandte Sozialwissenschaft GmbH, n.d.).
# To increase response rates and reduce sample selectivity, a mixed-mode survey design was used,
# combining paper-based postal questionnaire (PAPI), online survey (CAWI), and telephone interview (CATI).
# The figure shows the distribution of survey participants across the three interview modes.

interview_type_plot_data <- data_full$data |>
  filter(zeit %in% c(2018, 2021), !is.na(int_art)) |>
  mutate(
    zeit = factor(zeit),
    int_art = as.character(int_art),
    int_art_label = case_when(
      int_art == "1" ~ "CATI",
      int_art == "2" ~ "PAPI",
      int_art == "3" ~ "CAWI",
      TRUE ~ int_art
    )
  ) |>
  count(zeit, int_art_label)

interview_type_plot <- ggplot(
  interview_type_plot_data,
  aes(x = int_art_label, y = n, fill = zeit)
) +
  geom_col(position = position_dodge(width = 0.9)) +
  geom_text(
    aes(label = n),
    position = position_dodge(width = 0.9),
    vjust = -0.3,
    size = 3.5
  ) +
  labs(
    title = "Interview Types in 2018 and 2021",
    x = "Interview Type",
    y = "Count",
    fill = "Year"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  theme_ggplot

ggsave(
  filename = file.path(outputs_folder, "interview_types_2018_2021.png"),
  plot = interview_type_plot,
  width = 6,
  height = 4,
  dpi = dpi_ggplot
)
