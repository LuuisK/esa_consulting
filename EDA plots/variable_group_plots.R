source(file.path("utils", "saving_utils.R"))

data_discov <- load_data_bundle(
  file.path("data", "preprocessed", "data_discov_reduced")
)

data_valid <- load_data_bundle(
  file.path("data", "preprocessed", "data_valid_reduced")
)

vars_info <- readRDS(
  file.path("data", "preprocessed", "vars_info.rds")
)


library(ggplot2)
library(dplyr)


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

