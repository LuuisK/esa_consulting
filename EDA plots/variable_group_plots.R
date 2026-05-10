library(ggplot2)
library(dplyr)


source(file.path("utils", "saving_utils.R"))

data_discov <- load_data_bundle(
  file.path("data", "preprocessed", "data_discov_full")
)

data_valid <- load_data_bundle(
  file.path("data", "preprocessed", "data_valid_full")
)

vars_info <- readRDS(
  file.path("data", "preprocessed", "vars_info.rds")
)



# Anzahl Variablen pro Gruppe
p <- vars_info %>%
  count(group) %>%
  ggplot(aes(x = reorder(group, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Variables per Group",
    x = "",
    y = "Count"
  )

p

ggsave(
  filename = file.path(
    "outputs",
    "EDA_plots",
    "variables_per_group.png"
  ),
  plot = p,
  width = 8,
  height = 5,
  dpi = 300
)





# Demographic Structure of ESA Sample
# A) age distribution

ggplot(data_discov$data, aes(x = alter)) +
  geom_histogram(bins = 30) +
  labs(
    title = "Age Distribution",
    x = "Age",
    y = "Count"
  ) +
  theme_minimal()


# B) weighted age distribution
df <- data_discov$data %>%
  filter(!is.na(alter), !is.na(wichtn))

p <- ggplot(df, aes(x = alter, weight = wichtn)) +
  geom_histogram(bins = 30) +
  labs(
    title = "Weighted Age Distribution",
    x = "Age",
    y = "Weighted Count"
  ) +
  theme_minimal()

p












###### missingness before imputing skip NAs per group

###### missingness after imputing skip NAs per group


