# Graphic of weighted Clustersizes
# After running preprocessing.ipynb -> imputation.R -> dimensionality_reduction.R -> gmm_clustering.ipynb

# install packages
library(arrow)
library(ggplot2)
library(patchwork)

source(file.path("utils/saving_utils.R"))

min_n <- 100
output_file <- "outputs/clustering/plot_cluster_sizes.png"


# load data
load_wave <- function(wave) {
  pred <- read_parquet(file.path("outputs/gmm_clustering",
                                 paste0("cluster_pred_", wave, ".parquet")))
  bundle <- load_data_bundle(file.path("data/imputed",
                                       paste0("data_", wave, "_imputed")))
  stopifnot(nrow(pred) == nrow(bundle$data))
  data.frame(label = pred$label, wichtn = bundle$data$wichtn)
}


# weighted cluster sizes
make_sizes <- function(df, wave, prefix) {
  w <- tapply(df$wichtn, df$label, sum)
  w <- sort(w, decreasing = TRUE)
  data.frame(
    wave    = wave,
    cluster = paste0(prefix, seq_along(w) - 1),
    orig_id = names(w),
    n_w     = as.numeric(w),
    share   = 100 * as.numeric(w) / sum(w),
    small   = as.numeric(w) < min_n
  )
}

sizes <- rbind(make_sizes(load_wave("discov"), "discovery",  "D"),
               make_sizes(load_wave("valid"),  "validation", "V"))


# plot
plot_wave <- function(w, colour, title) {
  d <- subset(sizes, wave == w)
  d$cluster <- factor(d$cluster, levels = d$cluster)
  top5 <- round(sum(head(d$share, 5)))
  ymax <- ceiling(max(sizes$share) / 10) * 10

  ggplot(d, aes(x = cluster, y = share, fill = small)) +
    geom_col(width = 0.7) +
    geom_text(aes(label = sprintf("%.1f %%", share)),
              vjust = -0.5, size = 3, colour = "grey30") +
    scale_fill_manual(values = c(`FALSE` = colour, `TRUE` = "grey80"),
                      guide = "none") +
    scale_y_continuous(limits = c(0, ymax), breaks = seq(0, ymax, 10),
                       labels = function(x) paste0(x, " %"),
                       expand = expansion(mult = c(0, 0.02))) +
    labs(title = title,
         subtitle = paste0("five largest clusters: ", top5, " %"),
         x = "Cluster (numbered by size)", y = NULL) +
    theme_minimal(base_size = 13) +
    theme(panel.grid.major.x = element_blank(),
          panel.grid.minor   = element_blank(),
          plot.title    = element_text(face = "bold", colour = colour),
          plot.subtitle = element_text(colour = "grey40"))
}

p <- plot_wave("discovery",  "#00883A", "ESA 2018 (discovery)") +
  plot_wave("validation", "#1A1A80", "ESA 2021 (validation)")


print(p)
ggsave(output_file, p, width = 8, height = 3.5, dpi = 300, bg = "white")


# further indices
for (w in c("discovery", "validation")) {
  d   <- subset(sizes, wave == w)
  big <- d[!d$small, ]
  cat("\n", w, "\n",
      "Top 5:", round(sum(head(d$share, 5))), "%\n",
      sum(!d$small), "of", nrow(d), "clusters >=", min_n,
      "(", round(min(big$n_w)), "-", round(max(big$n_w)), ")\n",
      "Small:", paste0(d$cluster[d$small], " (", round(d$n_w[d$small]), ")",
                       collapse = ", "), "\n")
}
