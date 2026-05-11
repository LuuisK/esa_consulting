#' Script for creating scree plots

# Load required packages
library(ggplot2)
library(dplyr)
library(RColorBrewer)

source(file.path("utils/plotting_utils.R"))
source(file.path("utils/saving_utils.R"))

plots_folder <- file.path("outputs/dim_red")

default_colors <- theme_ggplot$palette.colour.discrete

mfa <- load_object(file.path("data/dim_red/mfa_discov.rds"))

explained_var <- as_tibble(mfa$eig, rownames = "component") |>
    rename(
        per_var = `percentage of variance`,
        cum_per_var = `cumulative percentage of variance`
    ) |>
    mutate(component = as.integer(sub("comp ", "", component))) |>
    filter(component <= 10)

coeff <- round(max(explained_var$cum_per_var) / max(explained_var$per_var))

scree_plot <- explained_var |>
    ggplot() +
    aes(x = as.factor(component)) +
    geom_col(aes(y = per_var, color = "per_var"), fill = default_colors[1]) +
    geom_line(aes(y = cum_per_var / coeff, group = 0, color = "cum_per_var")) +
    geom_point(
        aes(y = cum_per_var / coeff, group = 0, color = "cum_per_var"),
        shape = 4
    ) +
    scale_y_continuous(
        name = "percentage of variance",
        sec.axis = sec_axis(
            ~ coeff * .,
            name = "cumulative percentage of variance"
        )
    ) +
    scale_color_manual(
        name = "Measure",
        values = c(
            "per_var" = "transparent",
            "cum_per_var" = default_colors[2]
        ),
        labels = c(
            "per_var" = "Percentage of variance",
            "cum_per_var" = "Cumulative percentage\nof variance"
        )
    ) +
    labs(x = "component") +
    theme_ggplot +
    theme(panel.grid.major.x = element_blank())

ggsave(
    file.path(plots_folder, "scree_plot.png"),
    scree_plot,
    width = 4,
    height = 4,
    dpi = dpi_ggplot
)
