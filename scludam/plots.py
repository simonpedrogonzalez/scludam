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
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

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
    _, n_classes = proba.shape
    color_palette = sns.color_palette(palette, proba.shape[1])
    c = [color_palette[np.argmax(x)] for x in proba]
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
    c, proba_c, _ = color_from_proba(proba, palette)
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


def heatmap_of_detection_result_all_dimensions(
    self, peak: int = 0, mode: str = "hist", labels: str = None, x=0, y=1, **kwargs
):
    if self._last_result is None:
        raise ValueError("No result available, run detect() first")
    if self._last_result.centers.size == 0:
        raise ValueError("No peaks to plot")

    hist, edges = histogram(
          self._data,
          self.bin_shape,
          self._last_result.offsets[peak])
    pindex = self._last_result.indices[peak]

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from scludam.plots import heatmap

    dim = len(self.bin_shape)
    dims = np.arange(dim)

    if labels is None:
        labels = np.array([f"var{i+1}" for i in range(dim)], dtype="object")

    df = pd.DataFrame(self._data, columns=labels)

    g = sns.PairGrid(df)

    for i in range(dim):
        for j in range(dim):
            if i != j and i < j:
                # choose the dims to plot (x,y) and the dims to keep fixed
                xydims = dims[[i, j]]
                cutdims = np.array(list(set(dims) - set(xydims)))
                print("ploting", xydims, cutdims)

                # create a 2d cut for (x,y) with the other dims fixed
                # on the peak value
                cut = np.array([slice(None)] * dim, dtype="object")
                cut[cutdims] = pindex[cutdims]
                hist2D = hist[tuple(cut)]

                # get the edges of the 2d cut
                edges2D = np.array(edges, dtype="object")[xydims]
                # get the peak indices in the 2d cut
                pindex2D = pindex[xydims]
                if i > j:
                    pindex = np.flip(pindex)
                print("pindex2D", pindex2D)

                xticks = i == 0
                yticks = j == 0

                heatmap(
                    hist2D=hist2D,
                    edges=edges2D,
                    bin_shape=self.bin_shape,
                    index=pindex2D,
                    xticks=xticks,
                    yticks=yticks,
                    ax=g.axes[j, i],
                    **kwargs,
                )
            else:
                g.axes[j, i].set_visible(False)
    plt.show()
    print("coso")

def create_heatmaps(hist, edges, bin_shape, clusters_idx):
    dim = len(hist.shape)
    labels = [(np.round(edges[i] + bin_shape[i] / 2, 2))[:-1] for i in range(dim)]
    if dim == 2:
        data = hist
        annot_idx = clusters_idx
        annot = np.ndarray(shape=data.shape, dtype=str).tolist()
        for i in range(annot_idx.shape[1]):
            annot[annot_idx[0, i]][annot_idx[1, i]] = str(
                round(data[annot_idx[0][i]][annot_idx[1][i]])
            )
        ax = sns.heatmap(
            data, annot=annot, fmt="s", yticklabels=labels[0], xticklabels=labels[1]
        )
        hlines = np.concatenate((annot_idx[0], annot_idx[0] + 1))
        vlines = np.concatenate((annot_idx[1], annot_idx[1] + 1))
        ax.hlines(hlines, *ax.get_xlim(), color="w")
        ax.vlines(vlines, *ax.get_ylim(), color="w")
    else:
        cuts = np.unique(clusters_idx[2])
        ncuts = cuts.size
        ncols = min(3, ncuts)
        nrows = math.ceil(ncuts / ncols)
        delete_last = nrows > ncuts / ncols
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows,
                              figsize=(ncols * 8, nrows * 5))
        for row in range(nrows):
            for col in range(ncols):
                idx = col * nrows + row
                if idx < cuts.size:
                    cut_idx = cuts[idx]
                    data = hist[:, :, cut_idx]
                    annot_idx = clusters_idx.T[
                          (clusters_idx.T[:, 2] == cut_idx)
                          ].T[:2]
                    annot = np.ndarray(shape=data.shape, dtype=str).tolist()
                    for i in range(annot_idx.shape[1]):
                        annot[annot_idx[0, i]][annot_idx[1, i]] = str(
                            round(data[annot_idx[0][i]][annot_idx[1][i]])
                        )
                    if ncuts <= 1:
                        subplot = ax
                    else:
                        if nrows == 1:
                            subplot = ax[col]
                        else:
                            subplot = ax[row, col]
                    current_ax = sns.heatmap(
                        data,
                        annot=annot,
                        fmt="s",
                        yticklabels=labels[0],
                        xticklabels=labels[1],
                        ax=subplot,
                    )
                    current_ax.axes.set_xlabel("x")
                    current_ax.axes.set_ylabel("y")
                    # current_ax.title.set_text(
                    #     f"z slice at value {
                    #        round(edges[2][cut_idx]+bin_shape[2]/2, 4)
                    #       }"
                    # )
                    hlines = np.concatenate((annot_idx[0], annot_idx[0] + 1))
                    vlines = np.concatenate((annot_idx[1], annot_idx[1] + 1))
                    current_ax.hlines(hlines, *current_ax.get_xlim(), color="w")
                    current_ax.vlines(vlines, *current_ax.get_ylim(), color="w")
        if delete_last:
            ax.flat[-1].set_visible(False)
        fig.subplots_adjust(wspace=0.1, hspace=0.3)
        plt.tight_layout()
    return ax

def uniprobaplot_single(x, proba, statistic="median", bins=100, **kwargs):
    result = binned_statistic(x, proba, bins=bins, statistic=statistic)
    edges = result.bin_edges
    span_half = (edges[1] - edges[0]) / 2
    centers = edges[:-1] + span_half
    ax = sns.lineplot(x=centers, y=result.statistic, **kwargs)
    ax.fill_between(centers, result.statistic, alpha=0.5)
    return ax


def uniprobaplot(x, proba, **kwargs):
    ax = uniprobaplot_single(x, proba[:, 0], **kwargs)
    if not kwargs.get("ax"):
        kwargs["ax"] = ax
    for i in range(proba.shape[1] - 1):
        uniprobaplot_single(x, proba[:, i + 1], **kwargs)
    return ax

def membership_plot(
    data: Union[np.ndarray, pd.DataFrame],
    posteriors,
    labels=None,
    colnames: list = None,
    palette: str = "viridis",
    corner=True,  # breaks colorbar
    marker="o",
    alpha=0.5,
    density_intervals=4,
    density_mode="stack",
    size=None,
):

    sns.set_style("darkgrid")
    if isinstance(data, np.ndarray):
        obs, dims = data.shape
        data = pd.DataFrame(data)
        if colnames is not None:
            data.columns = colnames
        else:
            data.columns = [f"var {i+1}" for i in range(dims)]

    if labels is None and density_intervals is not None:
        if isinstance(density_intervals, int):
            density_intervals = np.linspace(0, 1, density_intervals + 1)
        labels = pd.cut(x=posteriors, bins=density_intervals, include_lowest=True)
        ticks = np.array([interval.right for interval in labels.categories.values])
        if density_mode == "stack":
            # reverse order in which stacked graf will appear
            hue_order = np.flip(labels.categories.astype(str))
            labels = labels.astype(str)
            if palette.endswith("_r"):
                diag_palette = palette.strip("_r")
            else:
                diag_palette = palette + "_r"
            diag_kws = {
                "hue": labels,
                "hue_order": hue_order,
                "multiple": density_mode,
                "palette": sns.color_palette(diag_palette, n_colors=len(hue_order)),
                "linewidth": 0,
                "cut": 0,
            }
        else:
            diag_kws = {
                "hue": labels,
                "multiple": density_mode,
                "palette": palette,
                "linewidth": 0,
                "cut": 0,
            }
    else:
        ticks = np.arange(0, 1, 0.1)
        diag_kws = {
            "hue": labels,
            "multiple": density_mode,
            "palette": palette,
            "linewidth": 0,
            "cut": 0,
        }

    sm = plt.cm.ScalarMappable(cmap=sns.color_palette(palette, as_cmap=True))
    sm.set_array([])

    plot_kws = {
        "hue": posteriors,
        "hue_norm": (0, 1),
        "marker": marker,
        "alpha": alpha,
        "palette": palette,
    }

    if size is not None:
        plot_kws["size"] = size

    grid = sns.pairplot(
        data,
        plot_kws=plot_kws,
        diag_kind="kde",
        diag_kws=diag_kws,
    )
    plt.colorbar(sm, ax=grid.axes, ticks=ticks)
    # its done this way to avoid map_lower error when having
    # different hues for diag and non diag elements
    if corner:
        for i in np.vstack(np.triu_indices(data.shape[1])).T:
            grid.axes[i[0], i[1]].set_visible(False)
    return grid


def membership_3d_plot(
    data: Union[np.ndarray, pd.DataFrame],
    posteriors,
    colnames: list = None,
    var_index=(1, 2, 3),
    palette: str = "viridis",
    alpha=0.5,
    marker="o",
    marker_size=40,
):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    cmap = ListedColormap(sns.color_palette(palette, 256).as_hex())
    if len(var_index) != 3:
        raise ValueError()
    var_index = np.array(var_index) - 1
    ax.scatter3D(
        data[:, var_index[0]],
        data[:, var_index[1]],
        data[:, var_index[2]],
        s=marker_size,
        marker=marker,
        cmap=cmap,
        c=posteriors,
        alpha=alpha,
    )
    return fig, ax

def plot3d_s(data, z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_trisurf(
        data[:, 0], data[:, 1], z, cmap="viridis", linewidth=0, antialiased=False
    )
    return ax

def probaplot(
    data: Union[np.ndarray, pd.DataFrame],
    proba,
    labels=None,
    colnames: list = None,
    palette: str = "viridis",
    corner=True,
    marker="o",
    multiple="layer",
    diag_kind="kde",
    size=None,
    alpha=1,
):

    sns.set_style("darkgrid")
    if isinstance(data, np.ndarray):
        obs, dims = data.shape
        data = pd.DataFrame(data)
        if colnames is not None:
            data.columns = colnames
        else:
            data.columns = [f"var {i+1}" for i in range(dims)]

    c, proba_c, color_palette = color_from_proba(proba, palette)

    diag_kws = {"hue": labels, "multiple": multiple, "palette": color_palette}

    plot_kws = {"marker": marker, "c": proba_c, "alpha": alpha}

    if size is not None:
        plot_kws["size"] = size

    grid = sns.pairplot(
        data,
        plot_kws=plot_kws,
        diag_kind=diag_kind,
        diag_kws=diag_kws,
    )

    # its done this way to avoid map_lower error when
    # having different hues for diag and non diag elements
    if corner:
        for i in np.vstack(np.triu_indices(data.shape[1])).T:
            grid.axes[i[0], i[1]].set_visible(False)

    return grid
