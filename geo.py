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
from sklearn.cluster import KMeans
import seaborn
import descartes
import cartopy.crs as ccrs


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
    df.info = df[df['region'] == 'JHM']

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

    for i, ax in enumerate(axes):
        plots[i].plot(ax=ax, markersize=3, color=colours[i])
        ctx.add_basemap(
                ax,
                crs=gdf.crs.to_string(),
                source=ctx.providers.Stamen.TonerLite,
                )
        gdf.boundary.plot(ax=ax, color="k")
        ax.axis("off")

    if fig_location:
        touch(fig_location)
        plt.savefig(fig_location)
    if show_figure:
        plt.show()


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru """
    gdf = gdf.to_crs("EPSG:3857")
    a = pd.Series(gdf['geometry'].apply(lambda p: p.x))
    b = pd.Series(gdf['geometry'].apply(lambda p: p.y))
    X = np.column_stack((a, b))
    n_clusters = 10

    # fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    # gdf.plot(column='cluster', ax=ax, markersize=3)

    kmeans = KMeans(n_clusters=n_clusters, random_state=5)
    labels = pd.Series(kmeans.fit_predict(X))

    gdf['cluster'] = labels.values
    # bin by cluster
    cluster_size: pd.DataFrame = gdf.groupby('cluster').cluster.count()

    # plot, using #circles / per cluster as the od weight
    clusters = kmeans.cluster_centers_

    print(type(clusters), clusters)
    print(type(np.array([cluster_size.to_numpy()])), np.array([cluster_size.to_numpy()]))

    clusters = np.hstack((clusters, np.array([cluster_size.to_numpy()]).transpose()))
    # np.append(clusters, np.array([cluster_size.to_numpy().transpose()]), axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    gdf.plot(ax=ax, markersize=1, color="tab:blue")
    plt.scatter(x=clusters[:, 0], y=clusters[:, 1],  # clusters x,y
                c=cluster_size,  # color
                s=clusters[:, 2] * 0.7,  # diameter, scaled
                facecolors='none',  # don't fill markers
                alpha=0.5)

    ctx.add_basemap(
            ax,
            crs=gdf.crs.to_string(),
            source=ctx.providers.Stamen.TonerLite,
            )
    gdf.boundary.plot(ax=ax, color="k")
    plt.colorbar()
    ax.axis("off")
    plt.show()

    if fig_location:
        touch(fig_location)
        plt.savefig(fig_location)
    if show_figure:
        plt.show()


if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    # plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)
