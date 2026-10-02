# Scree plots of the MFA for ESA 2018 (discovery) and ESA 2021 (validation)

library(dplyr)
library(ggplot2)

source(file.path("utils", "plotting_utils.R"))
source(file.path("utils", "saving_utils.R"))

# ------------------------------------------------------------------------------
# Load MFA results
# ------------------------------------------------------------------------------

mfa_discov <- load_object(file.path("data", "dim_red", "mfa_discov.rds"))
mfa_valid <- load_object(file.path("data", "dim_red", "mfa_valid.rds"))

outputs_folder <- file.path("outputs", "dim_red")
dir.create(outputs_folder, recursive = TRUE, showWarnings = FALSE)

n_comp <- 15   # dimensions used for clustering (dimensionality_reduction.R)
max_dim <- 30  # dimensions shown in the plots

default_colors <- theme_ggplot$palette.colour.discrete

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

save_plot <- function(plot, filename, width = 6, height = 4) {
  ggsave(
    filename = file.path(outputs_folder, filename),
    plot = plot,
    width = width,
    height = height,
    dpi = dpi_ggplot
  )
}

get_explained_var <- function(mfa) {
  as_tibble(mfa$eig, rownames = "dimension") |>
    rename(
      eigenvalue = eigenvalue,
      per_var = `percentage of variance`,
      cum_per_var = `cumulative percentage of variance`
    ) |>
    mutate(dimension = as.integer(sub("comp ", "", dimension))) |>
    filter(dimension <= max_dim)
}

explained_var <- bind_rows(
  get_explained_var(mfa_discov) |> mutate(year = "2018"),
  get_explained_var(mfa_valid) |> mutate(year = "2021")
) |>
  mutate(year = factor(year, levels = c("2018", "2021")))

cum_at_n <- explained_var |> filter(dimension == n_comp)
cum_label <- paste(
  sprintf("%.1f %% in %s", cum_at_n$cum_per_var, cum_at_n$year),
  collapse = ", "
)
x_breaks <- as.character(c(1, seq(5, max_dim, by = 5)))

# ------------------------------------------------------------------------------
# 1. Scree plot 2018 and 2021
# ------------------------------------------------------------------------------

coeff <- round(max(explained_var$cum_per_var) / max(explained_var$per_var))

scree_plot <- explained_var |>
  ggplot(aes(x = factor(dimension))) +
  geom_col(
    aes(y = per_var, fill = year),
    position = position_dodge(width = 0.9)
  ) +
  geom_line(aes(y = cum_per_var / coeff, colour = year, group = year)) +
  geom_point(aes(y = cum_per_var / coeff, colour = year), shape = 4) +
  geom_vline(xintercept = n_comp + 0.5, linetype = "dashed") +
  scale_fill_manual(values = colors_year) +
  scale_colour_manual(values = colors_year) +
  scale_x_discrete(breaks = x_breaks) +
  scale_y_continuous(
    name = "Explained variance (%)",
    expand = expansion(mult = c(0, 0.05)),
    sec.axis = sec_axis(~ . * coeff, name = "Cumulative explained variance (%)")
  ) +
  labs(
    title = "Scree Plot of the MFA in 2018 and 2021",
    subtitle = sprintf("Dashed line: %d dimensions used for clustering (%s)", n_comp, cum_label),
    x = "Dimension",
    fill = "Year",
    colour = "Year"
  ) +
  theme_ggplot

save_plot(
  scree_plot,
  "scree_plot_2018_2021.png",
  width = 7,
  height = 5
)

# ------------------------------------------------------------------------------
# 2. Scree plots per wave: discovery and validation
# ------------------------------------------------------------------------------

make_scree_plot <- function(data, title) {
  coeff_wave <- round(max(data$cum_per_var) / max(data$per_var))
  cum_wave <- data$cum_per_var[data$dimension == n_comp]

  data |>
    ggplot(aes(x = factor(dimension))) +
    geom_col(aes(y = per_var), fill = default_color) +
    geom_line(aes(y = cum_per_var / coeff_wave, group = 1), colour = default_colors[2]) +
    geom_point(aes(y = cum_per_var / coeff_wave), colour = default_colors[2], shape = 4) +
    geom_vline(xintercept = n_comp + 0.5, linetype = "dashed") +
    annotate(
      "text",
      x = n_comp + 0.7,
      y = cum_wave / coeff_wave,
      label = sprintf("%.1f %%", cum_wave),
      hjust = 0,
      vjust = 1.5,
      size = 3.5,
      family = font_name
    ) +
    scale_x_discrete(breaks = x_breaks) +
    scale_y_continuous(
      name = "Explained variance (%)",
      expand = expansion(mult = c(0, 0.05)),
      sec.axis = sec_axis(~ . * coeff_wave, name = "Cumulative explained variance (%)")
    ) +
    labs(
      title = title,
      subtitle = "Bars: variance per dimension · line: cumulative variance",
      x = "Dimension"
    ) +
    theme_ggplot
}

save_plot(
  make_scree_plot(filter(explained_var, year == "2018"), "Scree Plot of the MFA: Discovery"),
  "scree_plot_discovery.png",
  width = 6,
  height = 4
)

save_plot(
  make_scree_plot(filter(explained_var, year == "2021"), "Scree Plot of the MFA: Validation"),
  "scree_plot_validation.png",
  width = 6,
  height = 4
)

# ------------------------------------------------------------------------------
# 3. Values for the slides
# ------------------------------------------------------------------------------

write.csv(
  explained_var,
  file.path(outputs_folder, "mfa_explained_variance.csv"),
  row.names = FALSE
)

print(cum_at_n)