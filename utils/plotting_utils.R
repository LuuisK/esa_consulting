#' Module with utilities for plotting.

# Load required packages
library(ggplot2) # plotting
library(showtext)
library(sysfonts)


# Register font
font_name <- "Libertinus Serif"
font_add(
    family = font_name,
    regular = "fonts/Libertinus-7.051/LibertinusSerif-Regular.otf",
    bold = "fonts/Libertinus-7.051/LibertinusSerif-Bold.otf",
    italic = "fonts/Libertinus-7.051/LibertinusSerif-Italic.otf",
    bolditalic = "fonts/Libertinus-7.051/LibertinusSerif-BoldItalic.otf"
)
showtext_auto()
dpi_ggplot <- 300
showtext_opts(dpi = dpi_ggplot)


# Default pyplot colors
default_color <- "midnightblue"
colors_pyplot <- c(
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf"
)

# Create ggplot theme

# theme_ggplot <- theme_bw(base_family = font_name) +
#     theme(
#         geom = element_geom(color = default_color, fill = default_color),
#         palette.colour.discrete = colors_pyplot,
#         palette.fill.discrete = colors_pyplot
#     )
# scale_color_ggplot <- scale_color_hue()
# scale_fill_ggplot <- scale_fill_hue()



theme_ggplot <- theme_bw(base_family = font_name) +
    theme(
        text = element_text(color = default_color),
        axis.title = element_text(color = default_color),
        axis.text = element_text(color = default_color),
        plot.title = element_text(color = default_color, face = "bold")
    )

scale_color_ggplot <- scale_color_manual(values = colors_pyplot)
scale_fill_ggplot <- scale_fill_manual(values = colors_pyplot)

data.frame(x = c(0, 1, 2, 3), y = c(1, 1, 2, 2)) |>
    ggplot() +
    aes(x = x, y = y) +
    geom_line() +
    labs(x = "x", y = "y", title = "Test title") +
    theme_ggplot
