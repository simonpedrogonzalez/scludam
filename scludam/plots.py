import os
import sys
from copy import deepcopy
from typing import Union

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from attr import attrs, attrib, validators
from hdbscan import HDBSCAN, all_points_membership_vectors
from sklearn.base import ClassifierMixin, ClusterMixin, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    normalized_mutual_info_score,
    pairwise_distances,
)
import pandas as pd
from sklearn.preprocessing import RobustScaler
from astropy.table.table import Table
from astropy.stats.sigma_clipping import sigma_clipped_stats

from itertools import permutations
from scipy.stats import binned_statistic

sys.path.append(os.path.join(os.path.dirname("scludam"), "."))

# from scludam.hkde import HKDE, PluginSelector, pair_density_plot
# from scludam.synthetic import one_cluster_sample_small, three_clusters_sample
# from scludam.utils import combinations, Colnames
# from scludam.masker import RangeMasker, DistanceMasker, CrustMasker
# from scludam.plot_gauss_err import plot_kernels


# TODO: fix colors problem
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

    cmap = ListedColormap(sns.color_palette(palette).as_hex())
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
    # its done this way to avoid map_lower error when having different hues for diag and non diag elements
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


# is it useful?
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


def color_from_proba(proba, palette):
    color_palette = sns.color_palette(palette, proba.shape[1])
    c = [color_palette[np.argmax(x)] for x in proba]
    proba_c = [
        sns.desaturate(color_palette[np.argmax(x)], (np.max(x) - 0.5) / 0.5)
        for x in proba
    ]
    return c, proba_c, color_palette


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

    # its done this way to avoid map_lower error when having different hues for diag and non diag elements
    if corner:
        for i in np.vstack(np.triu_indices(data.shape[1])).T:
            grid.axes[i[0], i[1]].set_visible(False)

    return grid


def tsneplot(data, proba, palette="viridis", **kwargs):
    from sklearn.manifold import TSNE

    projection = TSNE().fit_transform(data)
    _, proba_c, _ = color_from_proba(proba, palette)
    return sns.scatterplot(projection[:, 0], projection[:, 1], c=proba_c, **kwargs)


def plot3d_s(data, z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_trisurf(
        data[:, 0], data[:, 1], z, cmap="viridis", linewidth=0, antialiased=False
    )
    return ax


def test_probaplot():
    data = np.random.uniform(0, 1, size=(1000, 2))
    p1 = data[:, 0]
    p2 = 1 - p1
    p = np.vstack((p1, p2)).T
    labels = np.zeros(1000)
    labels[p1 > 0.5] = 1
    probaplot(data, p, labels)
    plt.show()
    print("c0so")


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
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5))
        for row in range(nrows):
            for col in range(ncols):
                idx = col * nrows + row
                if idx < cuts.size:
                    cut_idx = cuts[idx]
                    data = hist[:, :, cut_idx]
                    annot_idx = clusters_idx.T[(clusters_idx.T[:, 2] == cut_idx)].T[:2]
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
                    current_ax.title.set_text(
                        f"z slice at value {round(edges[2][cut_idx]+bin_shape[2]/2, 4)}"
                    )
                    hlines = np.concatenate((annot_idx[0], annot_idx[0] + 1))
                    vlines = np.concatenate((annot_idx[1], annot_idx[1] + 1))
                    current_ax.hlines(hlines, *current_ax.get_xlim(), color="w")
                    current_ax.vlines(vlines, *current_ax.get_ylim(), color="w")
        if delete_last:
            ax.flat[-1].set_visible(False)
        fig.subplots_adjust(wspace=0.1, hspace=0.3)
        plt.tight_layout()
    return ax


def heatmap2D(hist2D: np.ndarray, edges: np.ndarray, bin_shape, index=None, annot=True, annot_prec=2, annot_threshold=.1, ticks=True, tick_prec=2, **kwargs):
    # annotations
    if annot == True:
        # create annotations as value of the histogram
        # but only for those bins that are above a certain threshold
        annot_indices = np.argwhere(hist2D.round(annot_prec) > annot_threshold)
        annot_values = hist2D[tuple(map(tuple, annot_indices.T))].round(annot_prec)
        annot = np.ndarray(shape=hist2D.shape, dtype=str).tolist()
        for i, xy in enumerate(annot_indices):
            annot[xy[0]][xy[1]] = str(annot_values[i])
        kwargs["annot"] = annot
        kwargs["fmt"]="s"
        annot_kws = kwargs.get("annot_kws", {})
        fontsize = annot_kws.get("fontsize", 8)
        annot_kws["fontsize"] = fontsize
        kwargs["annot_kws"] = annot_kws
        
    # labels
    # set tick labels as the value of the center of the bins, not the indices
    if ticks == True:
        labels = [np.round(edges[i] + bin_shape[i] / 2, tick_prec)[:-1] for i in range(2)]
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


def heatmap_slice(**kwargs):




# deprecated code sample
def heatmap_of_detection_result_all_dimensions(self, peak:int=0, mode:str="hist", labels:str=None, x=0, y=1, **kwargs):
        if self._last_result is None:
            raise ValueError("No result available, run detect() first")
        if self._last_result.centers.size == 0:
            raise ValueError("No peaks to plot")
        
        hist, edges = histogram(self._data, self.bin_shape, self._last_result.offsets[peak])
        pindex = self._last_result.indices[peak]
        
        from scludam.plots import heatmap
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt

        dim = len(self.bin_shape)
        dims = np.arange(dim)

        if labels is None:
            labels = np.array([f"var{i+1}" for i in range(dim)], dtype='object')

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
                    cut = np.array([slice(None)] * dim, dtype='object')
                    cut[cutdims] = pindex[cutdims]
                    hist2D = hist[tuple(cut)]

                    # get the edges of the 2d cut
                    edges2D = np.array(edges, dtype='object')[xydims]
                    # get the peak indices in the 2d cut
                    pindex2D = pindex[xydims]
                    if i > j:
                        pindex = np.flip(pindex)
                    print("pindex2D", pindex2D)
                    
                    xticks = i == 0
                    yticks = j == 0 
                    
                    heatmap(hist2D=hist2D, edges=edges2D, bin_shape=self.bin_shape, index=pindex2D, xticks=xticks, yticks=yticks, ax=g.axes[j, i], **kwargs)
                else:
                    g.axes[j,i].set_visible(False)
        plt.show()
        print('coso')