"""
Script for clustering.

Run scripts as interactive notebooks in the follwing order:
    preprocessing.py  ->  plot_corr_assoc.py  ->  dimensionality_reduction.py  ->  clustering.py
"""

import os
import pickle

import altair as alt
import numpy as np
import pandas as pd
import plotnine as p9
import prince
from matplotlib import colormaps
from sklearn.cluster import HDBSCAN, OPTICS

from utils import check_save_path

pd.options.mode.copy_on_write = True
alt.data_transformers.disable_max_rows()


# Hardcoded paths, change to apply to new data
famd_path = os.path.join("outputs", "esa_2018_2021_famd_alcohol.pkl")
data_reduced_path = os.path.join("outputs", "esa_2018_2021_reduced_alcohol.pkl")
data_path = os.path.join("data", "preprocessed", "esa_2018_2021_imputed.pkl")
vars_info_path = os.path.join("data", "preprocessed", "esa_vars_info.pkl")

image_folder = os.path.join("outputs", "images")
plot_save_path_template = os.path.join(image_folder, "clustering_plot_{}.png")


allow_overwrite = True


if __name__ == "__main__":
    # Read in data
    with open(famd_path, "rb") as file:
        famd: prince.FAMD = pickle.load(file)
    data_reduced: pd.DataFrame = pd.read_pickle(data_reduced_path)
    data: pd.DataFrame = pd.read_pickle(data_path)
    vars_info: pd.DataFrame = pd.read_pickle(vars_info_path)

    # Get group name
    famd_path_root, famd_path_ext = os.path.splitext(famd_path)
    group = famd_path_root.split(sep="_")[-1]
    group_vars = vars_info.loc[vars_info["group"] == group, "variable"]
    data = data[group_vars]

    # Run clustering
    # clustering = OPTICS(min_samples=5, metric='minkowski', p=2, metric_params=None, cluster_method='xi', xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', n_jobs=-1)
    clustering = HDBSCAN(min_cluster_size=100, metric="euclidean", metric_params=None, n_jobs=-1, store_centers="both")
    clustering.fit(data_reduced)

    cluster_labels = clustering.labels_
    # cluster_probs = clustering.probabilities_
    # cluster_centroids = clustering.centroids_
    # cluster_medoids = clustering.medoids_

    # Print information about clustering
    clusters = sorted(set(cluster_labels) - {-1})
    print(f"{len(clusters)} clusters, {100 * (cluster_labels == -1).mean():0.2f}% outliers")

    # Visualize clustering in the space of the two most important factors
    cmap = colormaps["Set1"]
    colors = [cmap(i) for i in range(len(clusters))]
    color_map = {cluster: color for cluster, color in enumerate(colors)}
    color_map[-1] = (0.6, 0.6, 0.6, 1.0)  # gray for outliers

    data_ggplot = data_reduced.rename(lambda x: f"comp_{x}", axis=1).assign(
        **{
            "cluster": pd.Categorical(
                [f"cluster {i}" if i >= 0 else np.nan for i in cluster_labels],
                categories=[f"cluster {i}" for i in clusters],
            )
        }
    )
    plot = (
        p9.ggplot(data_ggplot, p9.aes(x="comp_0", y="comp_1", fill="cluster"))
        + p9.geom_point(alpha=0.2)
        + p9.scale_fill_brewer(type="qualitative", na_value="white")
    )
    plot.show()
    plot_save_path = plot_save_path_template.format(group)
    check_save_path(plot_save_path, allow_overwrite=allow_overwrite)
    plot.save(plot_save_path)
    print(f"Saved plot of clustering as PNG to {plot_save_path}.")

    # data_to_plot = data_reduced.copy()
    # data_to_plot.columns = [f"component {i}" for i in data_to_plot.columns]
    # data_to_plot["cluster_label"] = pd.Categorical(cluster_labels)
    # plot = (
    #     alt.Chart(data_to_plot)
    #     .mark_circle(size=60)
    #     .encode(
    #         x=alt.X("component 0", type="quantitative", title="Factor1"),
    #         y=alt.Y("component 1", type="quantitative", title="Factor2"),
    #         color=alt.Color("cluster_label", type="nominal", scale=alt.Scale(scheme="set1")),
    #     )
    # )
    # plot.show()
