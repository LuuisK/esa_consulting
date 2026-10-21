# External variables by cluster:
# weighted age boxplots and one stacked bar chart per categorical variable
# (weighted shares among valid responses)

library(arrow)
library(dplyr)
library(ggplot2)
library(patchwork)
source("utils/saving_utils.R")

WAVES <- list(
  D = list(data = load_data_bundle("data/preprocessed/data_discov_full")$data,
           lab  = read_parquet("outputs/gmm_clustering/cluster_pred_discov.parquet")$label,
           title = "ESA 2018 — discovery", dark = "#3A8A45", light = "#D3E8D7"),
  V = list(data = load_data_bundle("data/preprocessed/data_valid_full")$data,
           lab  = read_parquet("outputs/gmm_clustering/cluster_pred_valid.parquet")$label,
           title = "ESA 2021 — validation", dark = "#151069", light = "#D2D3EA")
)
out_dir <- "outputs/clustering"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

num <- function(x) as.numeric(as.character(x))

# Map codes to categories; codes not listed (e.g. "no answer") become NA
recode_cat <- function(x, map) {
  out <- rep(NA_character_, length(x))
  for (nm in names(map)) out[num(x) %in% map[[nm]]] <- nm
  factor(out, levels = names(map))
}

# Categorical external variables (codes checked against the value labels).
# fam: the delivered value labels do not fit the data; read as
#      1 = married/registered partnership, 2 = never married, 3 = divorced, 4 = widowed.
# schule: 10 = no answer, excluded.
VARS <- list(
  gender      = \(d) recode_cat(d$ges, list(Men = 1, Women = 2)),
  citizenship = \(d) recode_cat(d$staat, list(German = 1, "German and other" = 3, Other = 2)),
  school      = \(d) recode_cat(d$schule, list("(Fach)abitur" = 5, "Mittlere Reife" = 3,
                                               Hauptschule = 2, "Other or none" = c(1, 6, 9))),
  employment  = \(d) recode_cat(d$erwerb, list("Full-time" = 1, "Part-time" = 2, Other = 3,
                                               "Not employed" = 4)),
  occupation  = \(d) recode_cat(d$Beruf, list(Worker = 1, "Employee, civil servant" = 2,
                                              "Self-employed" = 3, "In education or training" = 4,
                                              Other = 5)),
  income      = \(d) recode_cat(d$hne, list("< 1,500" = 1:5, "1,500 to < 2,500" = 6:9,
                                            "2,500 to < 4,000" = 10:11, "4,000 or more" = 12:13)),
  household   = \(d) recode_cat(pmin(num(d$hh), 4), list("1 person" = 1, "2" = 2, "3" = 3,
                                                         "4 or more" = 4)),
  children    = \(d) recode_cat(as.numeric(num(d$hh14) > 0), list("At least one" = 1, None = 0)),
  alone       = \(d) recode_cat(d$allein, list(Yes = 1, No = 0)),
  marital     = \(d) recode_cat(d$fam, list(Married = 1, "Never married" = 2, Divorced = 3,
                                            Widowed = 4))
)


#####################################################################################
# helpers
#####################################################################################

# Each respondent once in their cluster and once in "Total"
long_data <- function(p, value) {
  x <- tibble(value = value, w = num(WAVES[[p]]$data$wichtn), group = paste0(p, WAVES[[p]]$lab))
  bind_rows(x, mutate(x, group = "Total"))
}

# Weighted n and row position: clusters from the top, Total at the bottom; label "D0 (2,694.3)"
sizes <- function(p) {
  long_data(p, NA) |>
    group_by(group) |>
    summarise(n_w = sum(w)) |>
    mutate(pos = if_else(group == "Total", 0, 12 - suppressWarnings(as.numeric(substring(group, 2)))),
           lab = sprintf("%s (%s)", group, format(round(n_w, 1), nsmall = 1, big.mark = ",", trim = TRUE)))
}

# Common layout: row labels, separator line above Total, title in the wave colour
style <- function(p, sz) {
  list(geom_hline(yintercept = 1, colour = "grey75"),
       scale_y_continuous(breaks = sz$pos, labels = sz$lab, expand = expansion(add = 0.6)),
       labs(title = WAVES[[p]]$title, x = NULL, y = NULL),
       theme_minimal(base_size = 11),
       theme(plot.title = element_text(face = "bold", colour = WAVES[[p]]$dark, size = 13),
             axis.text.y = element_text(colour = "grey15", hjust = 0),
             panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
             plot.background = element_rect(fill = "white", colour = NA)))
}

# Save both waves side by side (slides) and one below the other (report)
save_both <- function(name, pD, pV, slide_height = 6) {
  ggsave(file.path(out_dir, sprintf("external_%s_slides.png", name)), pD | pV,
         width = 13.33, height = slide_height, dpi = 300)
  ggsave(file.path(out_dir, sprintf("external_%s_report.png", name)), pD / pV,
         width = 8, height = 9, dpi = 300)
}


#####################################################################################
# Stacked bars: weighted shares per category among valid responses
#####################################################################################

plot_bar <- function(p, f) {
  sz <- sizes(p)
  sh <- long_data(p, f(WAVES[[p]]$data)) |>
    filter(!is.na(value)) |>
    group_by(group, value) |>
    summarise(w = sum(w), .groups = "drop_last") |>
    mutate(share = 100 * w / sum(w), mid = cumsum(share) - share / 2) |>  # mid = label position
    ungroup() |>
    left_join(sz, by = "group")

  lv  <- levels(sh$value)
  pal <- setNames(colorRampPalette(c(WAVES[[p]]$dark, WAVES[[p]]$light))(length(lv)), lv)
  txt <- setNames(if_else(seq_along(lv) <= length(lv) / 2, "white", "grey15"), lv)

  ggplot(sh, aes(share, pos, fill = value)) +
    geom_col(orientation = "y", width = 0.7, position = position_stack(reverse = TRUE)) +
    geom_text(aes(x = mid, label = if_else(share >= 6, sprintf("%.1f %%", share), ""), colour = value),
              size = 3, fontface = "bold", show.legend = FALSE) +
    scale_fill_manual(values = pal, name = NULL) +
    scale_colour_manual(values = txt, guide = "none") +
    scale_x_continuous(expand = c(0, 0), breaks = NULL) +
    style(p, sz) +
    theme(legend.position = "top", legend.justification = "left",
          legend.key.size = unit(0.4, "cm"), panel.grid = element_blank())
}

for (key in names(VARS)) save_both(key, plot_bar("D", VARS[[key]]), plot_bar("V", VARS[[key]]))


#####################################################################################
# Age: weighted boxplots (box = weighted quartiles, line = weighted median,
# whiskers = most extreme ages within 1.5 x IQR, circles = outliers)
#####################################################################################
wq <- function(x, w, p) { o <- order(x); cw <- cumsum(w[o]) / sum(w); x[o][which(cw >= p)[1]] }

plot_age <- function(p) {
  sz <- sizes(p)
  d  <- long_data(p, num(WAVES[[p]]$data$alter)) |> filter(!is.na(value))
  st <- d |>
    group_by(group) |>
    summarise(q1 = wq(value, w, 0.25), med = wq(value, w, 0.5), q3 = wq(value, w, 0.75),
              lower = min(value[value >= q1 - 1.5 * (q3 - q1)]),
              upper = max(value[value <= q3 + 1.5 * (q3 - q1)])) |>
    left_join(sz, by = "group") |>
    mutate(dark  = if_else(group == "Total", "grey10", WAVES[[p]]$dark),
           light = if_else(group == "Total", "grey95", WAVES[[p]]$light))
  out <- d |> inner_join(st, by = "group") |> filter(value < lower | value > upper) |>
    distinct(pos, value, dark)

  ggplot(st, aes(y = pos, colour = dark)) +
    geom_segment(aes(x = lower, xend = upper, yend = pos)) +                            # whisker line
    geom_segment(aes(x = lower, xend = lower, y = pos - 0.15, yend = pos + 0.15)) +    # whisker caps
    geom_segment(aes(x = upper, xend = upper, y = pos - 0.15, yend = pos + 0.15)) +
    geom_rect(aes(xmin = q1, xmax = q3, ymin = pos - 0.3, ymax = pos + 0.3, fill = light)) +  # box
    geom_segment(aes(x = med, xend = med, y = pos - 0.3, yend = pos + 0.3), linewidth = 1.2) + # median
    geom_point(data = out, aes(x = value), shape = 21, fill = "white", size = 2) +     # outliers
    geom_text(aes(x = 67, label = med), colour = "grey10", fontface = "bold", size = 3.5, hjust = 0) +
    scale_colour_identity() + scale_fill_identity() +
    scale_x_continuous(breaks = seq(20, 60, 10), limits = c(18, 69)) +
    style(p, sz)
}

save_both("age", plot_age("D"), plot_age("V"), slide_height = 5.5)
