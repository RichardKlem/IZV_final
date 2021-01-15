#!/usr/bin/python3.8
# coding=utf-8
import os

import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import sklearn.cluster
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
import seaborn
import descartes
import cartopy.crs as ccrs

import scipy.stats
from scipy.stats import chi2_contingency

# muzeze pridat vlastni knihovny


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].apply(
            lambda x: pd.to_numeric(x, downcast="integer")
            )
    labels_to_categorize = [
            "p36",
            "p37",
            "weekday(p2a)",
            "p2b",
            "p6",
            "h",
            "i",
            "k",
            "p",
            "q",
            ]
    args = {}
    for label in labels_to_categorize:
        args[label] = "category"
    data_frame = df.astype(args)
    return data_frame


def touch(path):
    basedir = os.path.dirname(path)
    if not os.path.exists(basedir) and basedir != "":
        os.makedirs(basedir)

    with open(path, "a"):
        os.utime(path, None)


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovani"""
    df = optimize_dataframe(df)
    df.drop(df[np.isnan(df.d)].index, inplace=True)
    df.drop(df[np.isnan(df.e)].index, inplace=True)
    df = df[df['region'] == 'JHM']
    df.drop(df[df.d == -703607.94].index, inplace=True)  # Delete the outlier point.

    geo_df = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df["d"], df["e"]),
                                    crs="EPSG:5514")
    return geo_df


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ Vykresleni grafu s dvemi podgrafy podle lokality nehody """
    fig, axes = plt.subplots(1, 2, figsize=(20, 15))
    gdf = gdf.to_crs("EPSG:3857")

    gdf_inside = gdf[gdf["p5a"] == 1]
    gdf_outside = gdf[gdf["p5a"] == 2]
    plots = [gdf_inside, gdf_outside]
    colours = ["tab:orange", "tab:red"]
    titles = ['Nehody v Jihomoravském kraji v obci', 'Nehody v Jihomoravském kraji mimo obec']

    for i, ax in enumerate(axes):
        plots[i].plot(ax=ax, markersize=3, color=colours[i])
        ctx.add_basemap(
                ax,
                crs=gdf.crs.to_string(),
                source=ctx.providers.Stamen.TonerLite,
                )
        gdf.boundary.plot(ax=ax, color="k")
        ax.set_title(titles[i])
        ax.axis("off")

    if fig_location:
        touch(fig_location)
        plt.savefig(fig_location)
    if show_figure:
        plt.tight_layout()
        plt.show()


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru """
    gdf = gdf.to_crs("EPSG:3857")
    a = pd.Series(gdf['geometry'].apply(lambda p: p.x))
    b = pd.Series(gdf['geometry'].apply(lambda p: p.y))
    X = np.column_stack((a, b))
    n_clusters = 14

    kmeans = KMeans(n_clusters=n_clusters)
    labels = pd.Series(kmeans.fit_predict(X))

    gdf['cluster'] = labels.values
    cluster_size: pd.DataFrame = gdf.groupby('cluster').cluster.count()
    clusters: np.ndarray = kmeans.cluster_centers_

    clusters = np.hstack((clusters, np.array([cluster_size.to_numpy()]).transpose()))
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    gdf.plot(ax=ax, markersize=1, color="tab:blue")
    plt.scatter(x=clusters[:, 0],
                y=clusters[:, 1],
                c=cluster_size,
                s=clusters[:, 2] * 0.8,
                facecolors='none',
                alpha=0.6)
    ctx.add_basemap(
            ax,
            crs=gdf.crs.to_string(),
            source=ctx.providers.Stamen.TonerLite,
            )
    gdf.boundary.plot(ax=ax, color="k")
    color_bar = plt.colorbar(shrink=0.65)
    color_bar.set_alpha(1)
    color_bar.draw_all()
    ax.set_title('Nehody v Jihomoravském kraji')
    ax.axis("off")

    if fig_location:
        touch(fig_location)
        plt.savefig(fig_location)
    if show_figure:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)
