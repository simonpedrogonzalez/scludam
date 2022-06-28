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

import numpy as np
import seaborn as sns


def _heatmap2D(
    hist2D: np.ndarray,
    edges: np.ndarray,
    bin_shape,
    index=None,
    annot=True,
    annot_prec=2,
    annot_threshold=0.1,
    ticks=True,
    tick_prec=2,
    **kwargs,
):
    """Create a heatmap from a 2D histogram.

    Also marks index if provided. Create
    ticklabels from bin centers and not from
    bin indices.

    Parameters
    ----------
    hist2D : np.ndarray
        Histogram.
    edges : np.ndarray
        Edges.
    bin_shape : _type_
        Bin shape of the histogram.
    index : _type_, optional
        Index to be marked, by default None
    annot : bool, optional
        Use default annotations, by default True. If true,
        annotations are created taking into account the rest
        of annot parameters.
    annot_prec : int, optional
        Annotation number precision, by default 2
    annot_threshold : float, optional
        Only annotate cells with values bigger than
        annot_threshold, by default 0.1
    ticks : bool, optional
        Create ticklabels from the bin centers, by default True
    tick_prec : int, optional
        Ticklabels number precision, by default 2

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Plot. To get the figure from the result of the function,
        use ``fig = hist2d.get_figure()``.

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
