#' Script for creating plots about the variables

# Load required packages
library(ggplot2)
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
