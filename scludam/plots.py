# scludam, Star CLUster Detection And Membership estimation package
# Copyright (C) 2022  Simón Pedro González

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Module for helper plotting functions."""

from numbers import Number
from typing import List, Optional, Tuple, Union

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from scludam.type_utils import ArrayLike, Numeric1DArray, Numeric2DArray, NumericArray


def _prepare_data_to_plot(
    data: Union[Numeric2DArray, pd.DataFrame], cols: Optional[List[str]] = None
):
    if isinstance(data, np.ndarray):
        obs, dims = data.shape
        data = pd.DataFrame(data)
        if cols is not None:
            if len(cols) != dims:
                raise ValueError("Data and cols must have the same length.")
            data.columns = cols
        else:
            data.columns = [f"var {i+1}" for i in range(dims)]
    return data


def color_from_proba(proba: Numeric2DArray, palette: str):
    """Create color list from palette and probabilities.

    It desaturates the colors given the probabilities

    Parameters
    ----------
    proba : Numeric2DArray
        Membership probability array of shape
        (n_points, n_classes).
    palette : str
        Name of seaborn palette.

    Returns
    -------
    List
        Color list of length n_points where
        each point has a color according to the
        class it belongs.
    List
        Desaturated color list of length n_points
        where each point has a color according
        to the class it belongs. The saturation
        is higher if the probability is closer to
        1 and lower if it is closer to 1 / n_classes.
    List
        Color list of length n_classes, defining
        a color for each class.

    """
    if len(proba.shape) == 1:
        proba = np.atleast_2d(proba).T
    _, n_classes = proba.shape
    color_palette = sns.color_palette(palette, proba.shape[1])
    c = [color_palette[np.argmax(x)] for x in proba]

    if n_classes == 1:
        desaturation_factors = MinMaxScaler().fit_transform(proba)
        proba_c = [
            sns.desaturate(
                color_palette[0],
                des_fact,
            )
            for des_fact in desaturation_factors
        ]
    else:
        proba_c = [
            sns.desaturate(
                color_palette[np.argmax(x)],
                (np.max(x) - 1 / n_classes) / (1 - 1 / n_classes),
            )
            for x in proba
        ]
    return c, proba_c, color_palette


def scatter3dprobaplot(
    data: Union[Numeric2DArray, pd.DataFrame],
    proba: Numeric2DArray,
    cols: Optional[List[str]] = None,
    x: int = 0,
    y: int = 1,
    z: int = 2,
    palette: str = "viridis",
    desaturate: bool = True,
    **kwargs,
):
    """Create a 3D probability plot.

    It represents the provided data in x,
    y and z. It passes kwargs to matplotlib scatter3D [1]_

    Parameters
    ----------
    data : Union[Numeric2DArray, pd.DataFrame]
        Data to be plotted.
    proba : Numeric2DArray
        Array of membership probabilities, of shape
        (n_points, n_classes)
    cols : List[str], optional
        List of ordered column names, by default ``None``.
        Used if data is provided as numpy array.
    x : int, optional
        Index of the x variable, by default 0.
    y : int, optional
        Index of the y variable, by default 1.
    z : int, optional
        Index of the z variable, by default 2.
    palette : str, optional
        Seaborn palette string, by default "viridis"
    desaturate : bool, optional
        If ``True``, desaturate colors according to probability,
        by default ``True``.

    Returns
    -------
    matplotlib.collections.PathCollection
        Plot of the clustering results.

    Raises
    ------
    ValueError
        If data has less than 3 columns.

    References
    ----------
    .. [1] https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html?highlight=scatter3d#mpl_toolkits.mplot3d.axes3d.Axes3D.scatter3D

    """  # noqa: E501
    data = _prepare_data_to_plot(data, cols)
    cols = data.columns
    data = data.values
    if data.shape[1] < 3:
        raise ValueError("Data must have at least 3 columns.")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    c, proba_c, pal = color_from_proba(proba, palette)
    default_kws = {
        "c": proba_c if desaturate else c,
        "alpha": 1,
        "s": (proba.max(axis=1).round(2) * 100).astype(int) - 49,
    }
    default_kws.update(kwargs)
    ax.scatter3D(
        data[:, x],
        data[:, y],
        data[:, z],
        **default_kws,
    )
    ax.set_xlabel(cols[x])
    ax.set_ylabel(cols[y])
    ax.set_zlabel(cols[z])

    return fig, ax


def surfprobaplot(
    data: Union[pd.DataFrame, Numeric2DArray],
    proba: Numeric2DArray,
    x: int = 0,
    y: int = 1,
    palette: str = "viridis",
    cols: Optional[List[str]] = None,
    **kwargs,
):
    """Create surface 3D probability plot.

    It represents the provided data in x
    y. It passes kwargs to matplotlib plot_trisurf [2]_.

    Parameters
    ----------
    data : Union[pd.DataFrame, Numeric2DArray]
        Data to be plotted.
    proba : Numeric2DArray
        Membership probability array.
    x : int, optional
        Index of the x variable, by default 0
    y : int, optional
        Index of the y variable, by default 1
    palette : str, optional
        Seaborn palette string, by default "viridis"
    cols : List[str], optional
        List of ordered column names, by default ``None``.

    Returns
    -------
    matplotlib.collections.PathCollection
        Plot of the clustering results.

    Raises
    ------
    ValueError
        If data has less than 2 columns.
    ValueError
        If x or y parameters are invalid.

    References
    ----------
    .. [2] https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html?highlight=plot_trisurf#mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf

    """  # noqa: E501
    data = _prepare_data_to_plot(data, cols)
    cols = data.columns
    data = data.values
    if data.shape[1] < 2:
        raise ValueError("Data must have at least 2 columns.")
    if x == y or x >= data.shape[1] or y >= data.shape[1]:
        raise ValueError("Invalid x, y parameters.")
    if proba.shape[1] == 1:
        z = proba.ravel()
    else:
        z = proba[:, 1:].max(axis=1)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    default_kws = {
        "cmap": palette,
        "shade": True,
    }
    default_kws.update(kwargs)
    ax.plot_trisurf(
        data[:, x],
        data[:, y],
        z,
        **default_kws,
    )
    ax.set_xlabel(cols[x])
    ax.set_ylabel(cols[y])
    ax.set_zlabel("proba")
    return fig, ax


def pairprobaplot(
    data: Union[Numeric2DArray, pd.DataFrame],
    proba: Numeric2DArray,
    labels: Numeric1DArray,
    cols: Optional[List[str]] = None,
    palette: str = "viridis_r",
    diag_kind: str = "kde",
    diag_kws: Optional[dict] = None,
    plot_kws: Optional[dict] = None,
    **kwargs,
):
    """Pairplot of the data and the membership probabilities.

    It passes kwargs, diag_kws and plot_kws to seaborn pairplot [3]_
    function.

    Parameters
    ----------
    data : Union[Numeric2DArray, pd.DataFrame]
        Data to be plotted.
    proba : Numeric2DArray
        Membership probability array.
    labels : Numeric1DArray
        Labels of the data.
    cols : List[str], optional
        Column names, by default ``None``
    palette : str, optional
        Seaborn palette, by default "viridis_r"
    diag_kind : str, optional
        Kind of plot for diagonal, by default "kde".
        Valid values are "hist" and "kde".
    diag_kws : dict, optional
        Additional arguments for diagonal plots, by default ``None``
    plot_kws : dict, optional
        Additional arguments for off-diagonal plots, by default ``None``

    Returns
    -------
    seaborn.PairGrid
        Pairplot.

    Raises
    ------
    ValueError
        Invalid diag_kind.

    References
    ----------
    .. [3] https://seaborn.pydata.org/generated/seaborn.pairplot.html

    """
    df = _prepare_data_to_plot(data, cols)
    df["Label"] = labels.astype(str)
    hue_order = np.sort(np.unique(labels))[::-1].astype(str).tolist()
    df["Proba"] = proba.max(axis=1)

    if diag_kind == "kde":
        default_diag_kws = {
            "multiple": "stack",
            "fill": True,
            "linewidth": 0,
        }
    elif diag_kind == "hist":
        default_diag_kws = {
            "multiple": "stack",
            "element": "step",
            "linewidth": 0,
        }
    else:
        raise ValueError("Invalid diag_kind")

    if diag_kws is not None:
        default_diag_kws.update(diag_kws)

    default_plot_kws = {
        "marker": "o",
        "size": df["Proba"],
    }

    if plot_kws is not None:
        default_plot_kws.update(plot_kws)

    grid = sns.pairplot(
        df.loc[:, df.columns != "Proba"],
        hue="Label",
        hue_order=hue_order,
        palette=palette,
        diag_kind=diag_kind,
        diag_kws=default_diag_kws,
        plot_kws=default_plot_kws,
        **kwargs,
    )

    grid.legend.set_title("")

    return grid


def tsneprobaplot(
    data: Union[pd.DataFrame, Numeric2DArray],
    labels: Numeric1DArray,
    proba: Numeric2DArray,
    **kwargs,
):
    """Plot of data and membership probabilities using t-SNE projection.

    It pases kwargs to seaborn scatterplot [4]_ function.

    Parameters
    ----------
    data : Union[pd.DataFrame, Numeric2DArray]
        Data to be plotted.
    labels : Numeric1DArray
        Labels of the data.
    proba : Numeric2DArray
        Membership probability array.

    Returns
    -------
    matplotlib.axes.Axes
        T-SNE projected plot.

    References
    ----------
    .. [4] https://seaborn.pydata.org/generated/seaborn.scatterplot.html

    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    projection = TSNE().fit_transform(data)
    df = pd.DataFrame(
        {
            "x": projection[:, 0],
            "y": projection[:, 1],
            "Label": labels.astype(str),
            "Proba": proba.max(axis=1),
        }
    )
    default_kws = {
        "edgecolor": None,
        "size": df["Proba"],
        "palette": "viridis_r",
    }
    default_kws.update(kwargs)
    hue_order = np.sort(np.unique(labels))[::-1].astype(str).tolist()
    return sns.scatterplot(
        data=df, x="x", y="y", hue="Label", hue_order=hue_order, **default_kws
    )


def heatmap2D(
    hist2D: NumericArray,
    edges: ArrayLike,
    bin_shape: ArrayLike,
    index: ArrayLike = None,
    annot: bool = True,
    annot_prec: int = 2,
    annot_threshold: Number = 0.1,
    ticks: bool = True,
    tick_prec: int = 2,
    **kwargs,
):
    """Create a heatmap from a 2D histogram.

    Also marks index if provided. Create
    ticklabels from bin centers and not from
    bin indices. kwargs are passed to seaborn.heatmap [5]_.

    Parameters
    ----------
    hist2D : NumericArray
        Histogram.
    edges : ArrayLike
        Edges.
    bin_shape : ArrayLike
        Bin shape of the histogram.
    index : ArrayLike, optional
        Index to be marked, by default ``None``
    annot : bool, optional
        Use default annotations, by default ``True``. If true,
        annotations are created taking into account the rest
        of annot parameters.
    annot_prec : int, optional
        Annotation number precision, by default 2
    annot_threshold : Number, optional
        Only annotate cells with values bigger than
        annot_threshold, by default 0.1
    ticks : bool, optional
        Create ticklabels from the bin centers, by default ``True``
    tick_prec : int, optional
        Ticklabels number precision, by default 2

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Heatmap. To get the figure from the result of the function,
        use ``fig = heatmap2D.get_figure()``.

    References
    ----------
    .. [5] https://seaborn.pydata.org/generated/seaborn.heatmap.html

    """
    # annotations
    if annot:
        # create annotations as value of the histogram
        # but only for those bins that are above a certain threshold
        annot_indices = np.argwhere(hist2D.round(annot_prec) > annot_threshold)
        annot_values = hist2D[tuple(map(tuple, annot_indices.T))].round(annot_prec)
        if annot_prec == 0:
            annot_values = annot_values.astype(int)
        annot = np.ndarray(shape=hist2D.shape, dtype=str).tolist()
        for i, xy in enumerate(annot_indices):
            annot[xy[0]][xy[1]] = str(annot_values[i])
        kwargs["annot"] = annot
        kwargs["fmt"] = "s"
        annot_kws = kwargs.get("annot_kws", {})
        fontsize = annot_kws.get("fontsize", 8)
        annot_kws["fontsize"] = fontsize
        kwargs["annot_kws"] = annot_kws

    # labels
    # set tick labels as the value of the center of the bins, not the indices
    if ticks:
        labels = [
            np.round(edges[i] + bin_shape[i] / 2, tick_prec)[:-1] for i in range(2)
        ]
        kwargs["yticklabels"] = labels[0]
        kwargs["xticklabels"] = labels[1]

    if kwargs.get("cmap", None) is None:
        kwargs["cmap"] = "gist_yarg_r"

    hm = sns.heatmap(
        hist2D,
        **kwargs,
    )

    if index is not None:
        # add lines marking the peak
        hlines = [index[0], index[0] + 1]
        vlines = [index[1], index[1] + 1]
        hm.hlines(hlines, *hm.get_xlim(), color="w")
        hm.vlines(vlines, *hm.get_ylim(), color="w")
        hm.invert_yaxis()
        hm.set_xticklabels(hm.get_xticklabels(), rotation=45)
    return hm


def univariate_density_plot(
    x: Numeric1DArray,
    y: Numeric1DArray,
    ax: Optional[Axes] = None,
    figure: Optional[Figure] = None,
    figsize: Tuple[int, int] = (8, 6),
    grid: bool = True,
    **kwargs,
):
    """Plot univariate density plot.

    Create a filled lineplot given the
    densities for x. kwargs are passed
    to matplotlib scatter plot [6]_.

    Parameters
    ----------
    x : Numeric1DArray
        X linespace.
    y : Numeric1DArray
        Densities.
    ax : Optional[Axes], optional
        Ax to plot, by default None
    figure : Optional[Figure], optional
        Figure to plot, by default None
    figsize : Tuple[int, int], optional
        Figure size, by default (8, 6)
    grid : bool, optional
        Add grid, by default True

    Returns
    -------
    matplotlib.axes.Axes
        Axes of the plot.

    References
    ----------
    .. [6] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

    """
    if ax is None:
        if figure is None:
            figure = plt.figure(figsize=figsize)
        ax = figure.add_subplot(1, 1, 1)
    default_kwargs = {
        "color": "blue",
        "marker": ",",
        # "linestyle": "",
        # "lw": 0,
        # "linewidths": 0,
        "s": 0.01,
    }
    default_kwargs.update(kwargs)
    ax.scatter(x, y, **default_kwargs)
    zero = np.zeros(len(y))
    ax.fill_between(
        x,
        y,
        where=y >= zero,
        interpolate=True,
        color=default_kwargs.get("color", "blue"),
    )
    ax.set_yticks([], [])
    if grid:
        ax.grid("on")
    return ax


def bivariate_density_plot(
    x: Numeric1DArray,
    y: Numeric1DArray,
    z: Numeric1DArray,
    levels: int = None,
    contour_color: str = "black",
    ax: Optional[Axes] = None,
    figure: Optional[Figure] = None,
    figsize: Tuple[int, int] = (8, 6),
    colorbar: bool = True,
    title: Optional[str] = None,
    title_size: int = 16,
    grid: bool = True,
    **kwargs,
):
    """Create a bivariate density plot.

    Create a heatmap like density plot
    given densities in x and y. kwargs are
    passed to matplotlib imshow [7]_.

    Parameters
    ----------
    x : Numeric1DArray
        X linespace.
    y : Numeric1DArray
        Y linespace.
    z : Numeric1DArray
        Densities in x and y.
    levels : int, optional
        Number of levels to draw contour, by default None
    contour_color : str, optional
        Color to draw contour, by default "black"
    ax : Optional[Axes], optional
        Ax to plot, by default None
    figure : Optional[Figure], optional
        Figure to plot, by default None
    figsize : Tuple[int, int], optional
        Figure size, by default (8, 6)
    colorbar : bool, optional
        Add a colorbar, by default True
    title : Optional[str], optional
        Title to set, by default None
    title_size : int, optional
        Title size, by default 16
    grid : bool, optional
        Add grid, by default True

    Returns
    -------
    matplotlib.axes.Axes
        Axes of the plot.
    matplotlib.image.AxesImage
        Image of the plot.

    References
    ----------
    .. [7] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html

    """
    if ax is None:
        if figure is None:
            figure = plt.figure(figsize=figsize)
        ax = figure.add_subplot(1, 1, 1)

    if levels is not None:
        contour = ax.contour(x, y, z, levels, colors=contour_color)
        ax.clabel(contour, inline=True, fontsize=8)
        alpha = 0.75
    else:
        alpha = 1
    default_kws = {
        "origin": "lower",
        "aspect": "auto",
        "cmap": "inferno",
        "alpha": alpha,
    }
    default_kws.update(kwargs)
    im = ax.imshow(
        z,
        extent=[x.min(), x.max(), y.min(), y.max()],
        **default_kws,
    )
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax.get_figure().colorbar(im, cax=cax, orientation="vertical")
    if title is not None:
        ax.set_title(title, size=title_size)
    if grid:
        ax.grid("on")
    return ax, im


def add_label_legend(labels, palette: List[tuple], ax):
    patches = [
        mp.Patch(color=palette[i], label=f"Label {i}") for i in np.unique(labels)
    ]
    ax.legend(handles=patches)
    return ax


def scatter2dprobaplot(
    data, proba, labels, cols=None, palette="Set1", bg_kws: dict = {}, fr_kws: dict = {}
):
    if data.shape[1] != 2:
        raise ValueError("Data must have 2 columns")
    if isinstance(data, np.ndarray):
        if cols is not None:
            df = pd.DataFrame(data, columns=cols)
        else:
            df = pd.DataFrame(data, columns=["x", "y"])
    else:
        df = data
        if cols is not None:
            df.columns = cols
        else:
            cols = df.columns

    if proba.shape[0] != data.shape[0]:
        raise ValueError("proba must have the same number of rows as data")

    plotdf = pd.concat(
        [df, pd.DataFrame(np.max(proba, axis=1), columns=["Probability"])],
        axis=1,
        sort=False,
    )

    c, proba_c, label_c = color_from_proba(proba, palette)
    proba_c = np.array(proba_c)
    # plot background
    default_kws = {
        "s": 5,
        "alpha": 0.2,
        "color": label_c[0],
    }
    default_kws.update(bg_kws)
    ax = sns.scatterplot(data=plotdf[labels == -1], x=cols[0], y=cols[1], **default_kws)

    default_kws = {
        "sizes": (5, 50),
        "size": "Probability",
        "alpha": 0.8,
    }
    default_kws.update(fr_kws)

    for i in range(len(np.unique(labels)) - 1):
        sns.scatterplot(
            ax=ax,
            data=plotdf[labels == i],
            x=cols[0],
            y=cols[1],
            c=proba_c[labels == i],
            **default_kws,
        )

    # add_label_legend(labels, label_c, ax)
    return ax


def cm_diagram(
    data, proba, labels, cols=None, palette="Set1", bg_kws: dict = {}, fr_kws: dict = {}
):
    ax = scatter2dprobaplot(data, proba, labels, cols, palette, bg_kws, fr_kws)
    ax.invert_yaxis()
    ax.set_title("Color-Magnitude diagram")
    return ax


def scatter_with_coors(data, coors, palette="Paired", cols=["x", "y"], **kwargs):
    if len(coors.shape) == 1:
        coors = np.atleast_2d(coors)
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=cols)
    else:
        df = data
        cols = df.columns
    color_palette = sns.color_palette(palette, coors.shape[0])
    default_kws = {
        "marker": "o",
        "s": 10,
        "alpha": 0.5,
    }
    default_kws.update(kwargs)
    ax = sns.scatterplot(data=df, x=cols[0], y=cols[1], **default_kws)

    for i, c in enumerate(coors):
        ax.hlines(
            c[1], xmin=df[cols[0]].min(), xmax=df[cols[0]].max(), color=color_palette[i]
        )
        ax.vlines(
            c[0], ymin=df[cols[1]].min(), ymax=df[cols[1]].max(), color=color_palette[i]
        )

    return ax


# def patchscatter(data, masks, palette="Paired", cols=["x", "y"], **kwargs):
#     default_kws = {
#         "marker": "o",
#         "s": 5,
#         "alpha": 0.5,
#     }
#     default_kws.update(kwargs)
#     m1 = masks[0]
#     d1 = pd.DataFrame(data[m1], columns=cols)
#     ax = sns.scatterplot(d1[cols[0]], d1[cols[1]], **kwargs)
#     for m in masks[1:]:
#         d = pd.DataFrame(data[m], columns=cols)
#         sns.scatterplot(d[cols[0]], d[cols[1]], ax=ax, **kwargs)
#     return ax
