# Default pyplot colors
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
theme_ggplot <- theme_bw() +
    theme(
        palette.colour.discrete = colors_pyplot,
        palette.fill.discrete = colors_pyplot
    )
