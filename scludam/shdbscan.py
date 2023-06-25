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


"""Module for soft clustering numerical data using the HDBSCAN [1]_, [2]_ method.

This module provides a wrapper class for HDBSCAN [3]_ that adds some
extra functionality:

*   Calculation of probability-like scores from the soft clustering method.
*   Cluster selection based on geometric centers.
*   Custom plots to visualize the results.

References
----------
.. [1] McInnes, L., Healy, J., Astels, S. (2017). HDBSCAN: Hierarchical
    density based clustering. Journal of Open Source Software, 2, 11.
.. [2] Campello R.J.G.B., Moulavi D. y Sander J. (2013). Density-Based
    Clustering Based on Hierarchical Density Estimates. Advances in
    Knowledge Discovery and Data Mining, PAKDD 2013, Lecture Notes
    in Computer Science, 7819. doi: 10.1007/978-3-642-37456-2_14
.. [3] HDBSCAN: Hierarchical density-based spatial clustering of
    applications with noise. https://hdbscan.readthedocs.io/en/latest/

"""

from itertools import permutations
from typing import Callable, Optional, Union

import numpy as np
import seaborn as sns
from astropy.stats.sigma_clipping import sigma_clipped_stats
from attrs import define, field, validators
from beartype import beartype
from hdbscan import HDBSCAN, all_points_membership_vectors
from hdbscan.validity import validity_index
from sklearn.base import TransformerMixin
from sklearn.metrics import pairwise_distances

from scludam.plots import (
    pairprobaplot,
    scatter3dprobaplot,
    surfprobaplot,
    tsneprobaplot,
)
from scludam.type_utils import (
    Numeric1DArray,
    Numeric2DArray,
    Numeric2DArrayLike,
    OptionalNumeric1DArrayLike,
    OptionalNumeric2DArrayLike,
    _type,
)
from scludam.utils import one_hot_encode


@define
class SHDBSCAN:
    """Soft Hierarchical Density-Based Spatial Clustering of Applications with Noise.

    Class that wraps the HDBSCAN class to provide soft clustering
    calculations, cluster selection and custom plots.

    Attributes
    ----------
    min_cluster_size : Optional[int]
        Minimum size of cluster. Argument passed to HDBSCAN, by default ``None``.
        It is mandatory to provide this argument if the ``clusterer`` attribute
        is not provided.
    allow_single_cluster : bool
        Whether to allow single cluster or not, by default ``False``.
        Argument passed to HDBSCAN. In case that the data only
        contains one cluster and noise, the hierarchical clustering
        algorithm will not identify the cluster unless this option
        is set to ``True``.
    auto_allow_single_cluster : bool
        If ``True``, HDBSCAN will automatically toggle ``allow_single_cluster``
        to True if no clusters are found, to return at least 1 cluster. By
        default ``False``.
    min_samples : Optional[int]
        Minimum number of samples in a cluster, by default ``None``.
        Argument passed to HDBSCAN.
    metric : Union[str, Callable]
        Metric to be used in HDBSCAN, by default "euclidean".
    noise_proba_mode : str
        Method to calculate the noise probability, by default "score".
        Valid options are:

        *   score: Use only HDBSCAN cluster membership scores to calculate
            noise probability, as ``score = 1 - cluster_membership_score``,
            where ``cluster_membership_score`` is the HDBSCAN
            ``probabilities_`` value, which indicates how tied is
            a point to any cluster.
        *   outlier: Use ``scores`` as in the "score" option, and ``outlier_scores``
            to calculate the noise probability, as
            ``noise_proba = max(score, outlier_score)``.
            Outlier scores are calculated by HDBSCAN using the GLOSH [4]_ algorithm.
        *   conservative: Same method as the "outlier" option but does not
            allow for any point classified as noise to
            have a ``noise_proba`` lower than 1.

    cluster_proba_mode : str
        Method to calculate the cluster probability, by default "soft".
        Valid options are:

        *   soft: Use the HDBSCAN ``all_points_membership_vectors`` to calculate
            cluster probability, allowing for a point to be a member of multiple
            clusters.
        *   hard: Does not allow for a point to be a member of multiple clusters.
            A point can be considered noise or member of only one cluster.

    outlier_quantile : Optional[float]
        Quantile of outlier scores to be used as a threshold that defines a point
        as outlier, classified as noise, by default ``None``.
        It must be a value between 0 and 1. If provided,
        ``noise_proba_mode`` is set to "outlier".
        It scales HDBSCAN outlier scores so
        any point with an outlier score higher
        than the value of the provided quantile will
        be considered as noise.

    scaler : Optional[sklearn.base.TransformerMixin]
        Scaler to be used to scale the data before clustering, by default ``None``.

    clusterer : Optional[hdbscan.HDBSCAN [3]_]
        HDBSCAN clusterer to be used, by default ``None``. Used if more control is needed
        over the clustering algorithm. It is mandatory to provide this argument if
        the ``min_cluster_size`` attribute is not provided.

    n_classes : int
        Number of detected classes in the data sample. Only available after the
        :func:`~scludam.shdbscan.SHDBSCAN.fit` method is called.
    labels : Numeric1DArray
        Labels of the data sample. Only available after the
        :func:`~scludam.shdbscan.SHDBSCAN.fit` method is called. Noise
        points are labeled as -1, and the rest of the points are labeled
        with the cluster index.
    proba : Numeric2DArray
        Probability of each point to belog to each class, including.
        Only available after the
        :func:`~scludam.shdbscan.SHDBSCAN.fit` method is called. Array of shape
        ``(n_samples, n_classes)``. The first column corresponds to the noise class.
    outlier_scores : Numeric1DArray
        Outlier scores of each point. Only available after the
        :func:`~scludam.shdbscan.SHDBSCAN.fit` method is called.

    Raises
    ------
    ValueError
        If the ``min_cluster_size`` nor the ``clusterer`` attributes are provided.


    Examples
    --------
    .. literalinclude:: ../../examples/shdbscan/shdbscan.py
        :language: python
        :linenos:
    .. image:: ../../examples/shdbscan/shdbscan_pairplot.png

    References
    ----------
    .. [4] https://hdbscan.readthedocs.io/en/latest/outlier_detection.html?highlight=glosh#outlier-detection

    """  # noqa: E501

    # input attrs
    min_cluster_size: Optional[int] = field(
        default=None, validator=_type(Optional[int])
    )
    auto_allow_single_cluster: bool = field(default=False, validator=_type(bool))
    allow_single_cluster: bool = field(default=False, validator=_type(bool))
    min_samples: Optional[int] = field(default=None, validator=_type(Optional[int]))
    metric: Union[str, Callable] = field(default="euclidean")
    noise_proba_mode: str = field(
        default="score",
        validator=[_type(str), validators.in_(["score", "outlier", "conservative"])],
    )
    cluster_proba_mode: str = field(
        default="soft", validator=[_type(str), validators.in_(["soft", "hard"])]
    )
    outlier_quantile: Optional[float] = field(
        default=None,
        validator=[
            _type(Optional[float]),
            validators.optional([validators.ge(0), validators.le(1)]),
        ],
    )
    scaler: Optional[TransformerMixin] = field(
        default=None, validator=_type(Optional[TransformerMixin])
    )
    clusterer: Optional[HDBSCAN] = field(
        default=None, validator=_type(Optional[HDBSCAN])
    )

    # internal attrs
    _data: Numeric2DArray = None
    _centers_provided: bool = False
    _centers: OptionalNumeric2DArrayLike = None
    _n: int = None
    _d: int = None
    _unique_labels: np.ndarray = None

    # output attrs
    n_classes: int = None
    proba: Numeric2DArray = None
    labels: Numeric1DArray = None
    outlier_scores: Numeric1DArray = None

    @min_cluster_size.validator
    def _min_cluster_size_validator(self, attribute, value):
        if value is None:
            if self.clusterer is None:
                raise ValueError(
                    "Either min_cluster_size or clusterer must be provided."
                )
        elif value < 1:
            raise ValueError("min_cluster_size must be greater than 1.")

    def _cluster(self, data: Numeric2DArray):
        if self._centers is not None or self.auto_allow_single_cluster:
            allow_single_cluster = False
        else:
            allow_single_cluster = self.allow_single_cluster

        if self.clusterer is None:
            if self.min_samples is None:
                min_samples = self.min_cluster_size
            else:
                min_samples = self.min_samples

            self.clusterer = HDBSCAN(
                min_samples=min_samples,
                min_cluster_size=self.min_cluster_size,
                allow_single_cluster=allow_single_cluster,
                metric=self.metric,
                prediction_data=True,
            )
        else:
            self.clusterer.allow_single_cluster = allow_single_cluster
            self.clusterer.prediction_data = True

        self.clusterer.fit(data)

        unique_labels = np.sort(np.unique(self.clusterer.labels_))

        # if no clusters found & auto_toggle_single_cluster & not allow_single_cluster
        if np.all(unique_labels == -1) and self.auto_allow_single_cluster:
            self.clusterer = HDBSCAN(
                min_samples=self.clusterer.min_samples,
                min_cluster_size=self.clusterer.min_cluster_size,
                allow_single_cluster=True,
                metric=self.clusterer.metric,
                prediction_data=True,
            )
            self.clusterer.fit(data)

        return self.clusterer

    # get the most "conservative" cluster probabilities
    # accounting for outlier scores and labeling given
    # by the clusterer
    def _get_proba(self):
        # Outlier score is an implementation of GLOSH, it catches local outliers as
        # well as just points that are far away from everything.
        # Thus a point can be "in"
        # a cluster, and have a label, but be sufficiently far from an
        # otherwise very
        # dense core that is is anomalous in that local region of space
        # (i.e. it is weird
        # to have a point there when almost everything else is far more
        # tightly grouped).

        # The probabilties are slightly misnamed. It is essentially a
        # "cluster membership score"
        # that is, if the point is in a cluster how relatively well tied to the
        # cluster is it?
        # It is effectively the ratio of the distance scale at which this point
        # fell out of the cluster
        # with the distance scale of the core of the cluster.

        # Any noise points are assigned a probability 0 as it is the
        # "membership score"
        # for the cluster that they are a member of, and noise points are not
        # a member of any cluster.

        # https://github.com/scikit-learn-contrib/hdbscan/issues/80

        one_hot_code = one_hot_encode(self.clusterer.labels_)
        # in case no noise points are found, add a column of zeros for noise
        if not np.any(self._unique_labels == -1):
            one_hot_code = np.vstack(
                (np.zeros(one_hot_code.shape[0]), one_hot_code.T)
            ).T

        n, n_classes = one_hot_code.shape
        n_clusters = n_classes - 1

        # get sanitized outlier scores
        outlier_scores = np.copy(self.clusterer.outlier_scores_)
        outlier_scores[outlier_scores == -np.inf] = 0
        outlier_scores[outlier_scores == np.inf] = 1

        # if quantile is used, set mode to "outlier" and
        # scale the outlier scores considering that
        # scores > quantile(scores, q) are outliers, so
        # new score is 1 for them, and the others get
        # rescaled with max = quantile(scores, q) and min = 0
        if self.outlier_quantile is not None:
            # self.outlier_quantile is a float between 0 and 1
            # self.outlier_quantile = 0 == self.outlier_quantile = None
            # use quantile to determine scale outlier score
            # if we consider outlier for score > quantile
            # then we scale de outlier score with that threshold
            # as being the defining point when p(outlier) = 1
            top = np.quantile(outlier_scores, self.outlier_quantile)
            if top == 0:
                raise ValueError(
                    "outlier_quantile selected is too low, the value for it is 0."
                )
            outlier_scores[outlier_scores > top] = top
            outlier_scores /= top
            # last line is equivalent to
            # outlier_scores = (
            #     MinMaxScaler().fit_transform(outlier_scores.reshape(-1, 1)).ravel()
            # )
            # considering that we want data in 0,1 interval and that
            # we want to use 0 as min, instead of outlier_scores.min()
            # as the inf limit for outlier_scores should be fixed in 0
            # no mather the values of the outlier_scores.
            self.noise_proba_mode = "outlier"

        self.outlier_scores = outlier_scores
        noise_score_arrays = [1 - self.clusterer.probabilities_]

        if self.noise_proba_mode == "outlier":
            # take into account outlier_scores as indicative
            # of noise
            noise_score_arrays.append(outlier_scores)

        if self.noise_proba_mode == "conservative":
            # take into account labels
            # so no point classified as noise can
            # have any cluster probability
            noise_score_arrays.append(one_hot_code[:, 0])

        noise_proba = np.vstack(noise_score_arrays).max(0)

        # array with the repeated sum of cluster proba to multiply
        cluster_proba_sum = np.tile((1 - noise_proba), (n_clusters, 1)).T

        if (
            not self.clusterer.allow_single_cluster
            and self.cluster_proba_mode == "soft"
        ):
            # can calculate membership vectors and soft mode selected
            membership = all_points_membership_vectors(self.clusterer)
            # calculate membership only considering clusters, no noise
            only_cluster_membership = np.zeros_like(membership)
            non_zero_cluster_mem = ~np.isclose(self.clusterer.probabilities_, 0)
            only_cluster_membership[non_zero_cluster_mem] = membership[
                non_zero_cluster_mem
            ] / self.clusterer.probabilities_[non_zero_cluster_mem][
                :, np.newaxis
            ].repeat(
                n_clusters, axis=1
            )
            # scale membership taking noise into account so noise + cluster = 1
            # if noise_proba = 1 - probabilities_ (mode=score)
            # then this is not necessary but the result is nevertheless correct.
            cluster_proba = only_cluster_membership * cluster_proba_sum

            # check for the possible errors in membership vector
            # and fix them. This error should not happen often
            # but can happen.
            # This part of the code can be removed when hdbscan is fixed
            # https://github.com/scikit-learn-contrib/hdbscan/issues/246
            # the probability diff is absorbed by the cluster_proba
            # because the origin of the error is in the cluster membership
            has_error = ~np.isclose(noise_proba + cluster_proba.sum(axis=1), 1)
            if np.any(has_error):
                diff = 1 - (
                    noise_proba[has_error] + cluster_proba[has_error].sum(axis=1)
                )
                diff_per_cluster = diff / n_clusters
                cluster_proba[has_error] += diff_per_cluster
        else:
            # can't calculate membership vectors, or hard mode selected
            # in this mode, no point can have a mixture of cluster proba
            # so it is classifying the points as either one cluster
            # or another.
            cluster_proba = one_hot_code[:, 1:] * cluster_proba_sum

        if len(cluster_proba.shape) == 1:
            cluster_proba = np.atleast_2d(cluster_proba).T

        proba = np.empty((n, n_classes))
        proba[:, 0] = noise_proba
        proba[:, 1:] = cluster_proba
        assert np.allclose(proba.sum(axis=1), 1)
        return proba

    def _center_based_cluster_selection(
        self,
        data: Numeric2DArray,
        labels: Numeric1DArray,
        input_centers: Numeric2DArrayLike,
    ):
        # compares input cluster centers with obtained cluster centers
        # if input cluster centers are less than obtained, then select
        # onbtained clusters that match input centers the best

        cluster_labels = self._unique_labels[self._unique_labels != -1]
        cluster_centers = np.array(
            [
                [
                    sigma_clipped_stats(
                        data[labels == label][:, i],
                        cenfunc="median",
                        stdfunc="mad_std",
                        maxiters=None,
                        sigma=1,
                    )[1]
                    for i in range(self._d)
                ]
                for label in cluster_labels
            ]
        )

        # there are obtained clusters to label as noise
        # we should select those that match input centers the best

        short = input_centers
        long = cluster_centers

        center_distances = pairwise_distances(X=short, Y=long)
        idx_columns = np.array(
            list(permutations(np.arange(long.shape[0]), short.shape[0]))
        )
        idx_rows = np.arange(short.shape[0])

        if short.shape[0] == 1:
            distance_sum_per_solution = center_distances.ravel()
        else:
            dist_idcs = tuple(
                [
                    tuple(map(tuple, x))
                    for x in np.stack(
                        (np.tile(idx_rows, (idx_columns.shape[0], 1)), idx_columns),
                        axis=1,
                    )
                ]
            )
            distance_sum_per_solution = np.array(
                [center_distances[dist_idcs[i]] for i in range(len(dist_idcs))]
            ).sum(axis=1)

        best_solution = idx_columns[
            distance_sum_per_solution == distance_sum_per_solution.min()
        ].ravel()

        # lets delete some clusters
        # shot is self.class_centers
        # long is class_centers
        # labels are in class_centers order
        # i need to keep labels that are in best_solution
        # the rest should be noise

        new_labels = np.copy(labels)
        new_labels[~np.isin(labels, best_solution)] = -1

        posteriors = self._get_proba()

        noise_proba = posteriors[
            :,
            tuple(
                [0]
                + list((cluster_labels + 1)[~np.isin(cluster_labels, best_solution)])
            ),
        ].sum(axis=1)
        cluster_proba = posteriors[
            :, tuple((cluster_labels + 1)[np.isin(cluster_labels, best_solution)])
        ]

        new_n_classes = short.shape[0] + 1

        # create new posteriors array
        new_posteriors = np.zeros((self._n, new_n_classes))
        new_posteriors[:, 0] = noise_proba
        new_posteriors[:, 1:] = cluster_proba

        assert np.allclose(new_posteriors.sum(axis=1), 1)

        # reorder kept labels
        for i, label in enumerate(best_solution):
            new_labels[new_labels == label] = i

        return new_labels, new_posteriors

    def _scale(self, data: Numeric2DArray, centers: OptionalNumeric2DArrayLike):
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        if self._centers_provided:
            centers = self.scaler.transform(centers)
        return data, centers

    @beartype
    def fit(
        self,
        data: Numeric2DArray,
        centers: Union[OptionalNumeric2DArrayLike, OptionalNumeric1DArrayLike] = None,
    ):
        """Fit the clusterer to the data.

        It uses the provided configuration to identify clusters,
        classify the data and provide membership probabilities. The results
        are stored in the :class:`~scludam.shdbscan.SHDBSCAN` instance. The
        attributes storing results are :attr:`~scludam.shdbscan.SHDBSCAN.n_classes`,
        :attr:`~scludam.shdbscan.SHDBSCAN.labels`,
        :attr:`~scludam.shdbscan.SHDBSCAN.proba` and
        :attr:`~scludam.shdbscan.SHDBSCAN.outlier_scores`.

        Parameters
        ----------
        data : Numeric2DArray
            Data to be clustered.
        centers : Union[Numeric2DArrayLike, Numeric1DArrayLike], optional
            Center or array of centers of clusters, by default ``None``.
            If provided,
            only the clusters that are geometrically closer to the
            provided centers will be considered. This option is useful for
            ignoring clusters in a multiple cluster scenario.

        Returns
        -------
        SHDBSCAN
            Fitted instance of the :class:`~scludam.shdbscan.SHDBSCAN` class.

        Raises
        ------
        ValueError
            If the dimensions of the centers array do not match the dimensions
            of the data array.

        """
        self._data = np.atleast_2d(np.asarray(data))

        if centers is not None:
            self._centers = np.atleast_2d(np.asarray(centers))
            if self._centers.shape[0] <= 0:
                self._centers_provided = False
            else:
                if self._centers.shape[1] != self._data.shape[1]:
                    raise ValueError(
                        "Provided centers have different number of dimensions than data"
                    )
                else:
                    self._centers_provided = True
        else:
            self._centers_provided = False

        self._n, self._d = self._data.shape

        if self.scaler:
            self._data, self._centers = self._scale(self._data, self._centers)

        self._cluster(self._data)

        self.labels = self.clusterer.labels_
        self._unique_labels = np.sort(np.unique(self.labels))
        self.n_classes = self._unique_labels.shape[0]

        # case only noise or only 1 cluster with no noise
        if self.n_classes == 1:
            self.proba = one_hot_encode(self.labels)
            return self

        n_clusters = self._unique_labels[self._unique_labels != -1].shape[0]
        # case cluster selection required
        if (
            not self.clusterer.allow_single_cluster
            and self._centers_provided
            and self._centers.shape[0] < n_clusters
        ):
            self.labels, self.proba = self._center_based_cluster_selection(
                self._data, self.labels, self._centers
            )
        else:
            self.proba = self._get_proba()
            self.labels = np.argmax(self.proba, axis=1) - 1

        self._unique_labels = np.sort(np.unique(self.labels))
        self.n_classes = self._unique_labels.shape[0]

        if not np.any(self._unique_labels == -1):
            # noise label should always be present even if there is no noise
            # because probability column is included
            self._unique_labels = np.array([-1] + list(self._unique_labels))

        return self

    def _is_fitted(self):
        return (
            self.labels is not None
            and self.proba is not None
            and self._data is not None
        )

    def validity_index(self, **kwargs):
        """Compute the validity index of the clustering.

        Calculates HDBSCAN density validity index [5]_ for the
        labels obtained from the clustering. ``kwargs`` are passed to
        the HDBSCAN ``validity_index`` method.

        Returns
        -------
        float
            Density based cluster validity index between -1 and 1. A
            higher value means a better clustering.
        Numeric1DArray
            Array of cluster validity indices for each cluster, only if
            ``per_cluster_scores`` kwarg is set to True.

        Raises
        ------
        Exception
            If the clustering has not been performed yet.

        References
        ----------
        .. [5] https://hdbscan.readthedocs.io/en/latest/api.html?highlight=validity_index#hdbscan.validity.validity_index

        """  # noqa: E501
        if not self._is_fitted():
            raise Exception("Clusterer not fitted. Try excecuting fit function first.")
        return validity_index(self._data, self.labels, **kwargs)

    def pairplot(self, **kwargs):
        """Plot the clustering results in a pairplot.

        It uses the :func:`~scludam.plots.pairprobaplot`. The colors
        of the points represent class labels. The sizes
        of the points reresent the probability of belonging to
        the most probable class.

        Returns
        -------
        seaborn.PairGrid
            Pairplot of the clustering results.

        Raises
        ------
        Exception
            If the clustering has not been performed yet.

        """
        if not self._is_fitted():
            raise Exception("Clusterer not fitted. Try excecuting fit function first.")
        return pairprobaplot(
            data=self._data, proba=self.proba, labels=self.labels, **kwargs
        )

    def tsneplot(self, **kwargs):
        """Plot the clustering results in a t-SNE plot.

        It uses the :func:`~scludam.plots.tsneprobaplot` function.
        It represents the data in a 2 dimensional space using t-SNE.
        The colors of the points represent class labels.
        The sizes of the points represent the probability of belonging
        to the most probable class.

        Returns
        -------
        matplotlib.axes.Axes
            Plot of the clustering results.

        Raises
        ------
        Exception
            If the clustering has not been performed yet.

        """
        if not self._is_fitted():
            raise Exception("Clusterer not fitted. Try excecuting fit function first.")
        return tsneprobaplot(
            data=self._data, proba=self.proba, labels=self.labels, **kwargs
        )

    def scatter3dplot(self, **kwargs):
        """Plot the clustering results in a 3D scatter plot.

        It uses the :func:`~scludam.plots.scatter3dprobaplot` function.
        It represents the data in a 3 dimensional space using the
        variables given by the user. The colors of the points
        represent class labels. The sizes of the points represent
        the probability of belonging to the most probable class.

        Returns
        -------
        matplotlib.collections.PathCollection
            Plot of the clustering results.

        Raises
        ------
        Exception
            If the clustering has not been performed yet.

        """
        if not self._is_fitted():
            raise Exception("Clusterer not fitted. Try excecuting fit function first.")
        return scatter3dprobaplot(self._data, self.proba, **kwargs)

    def surfplot(self, **kwargs):
        """Plot the clustering results in a 3D surface plot.

        It uses the :func:`~scludam.plots.surfprobaplot` function.
        The heights of the surface and colors of the points represent
        the probability of belonging to the most probable cluster,
        excluding the noise class. The data is represented in two
        dimensions, given by the user.

        Returns
        -------
        matplotlib.collections.PathCollection
            Plot of the clustering results.

        Raises
        ------
        Exception
            If the clustering has not been performed yet.

        """
        if not self._is_fitted():
            raise Exception("Clusterer not fitted. Try excecuting fit function first.")
        return surfprobaplot(self._data, self.proba, **kwargs)

    def outlierplot(self, **kwargs):
        """Plot the distribution of outlier scores.

        Includes an indicator of ``outlier_quantile`` if provided.
        It is useful for choosing an appropriate value for
        ``outlier_quantile``. Uses seaborn displot function [6]_.

        Returns
        -------
        matplotlib.axes.Axes
            Plot of the outlier scores distribution.

        Raises
        ------
        Exception
            If the clustering has not been performed yet.

        Examples
        --------
        .. literalinclude:: ../../examples/shdbscan/outlierplot.py
            :language: python
            :linenos:
        .. image:: ../../examples/shdbscan/outlierplot.png

        References
        ----------
        .. [6] https://seaborn.pydata.org/generated/seaborn.distplot.html?highlight=distplot#seaborn.distplot

        """  # noqa E501
        if not self._is_fitted():
            raise Exception("Clusterer not fitted or no outlier were calculated.")
        outlier_scores = self.clusterer.outlier_scores_[
            np.isfinite(self.clusterer.outlier_scores_)
        ]
        rug = kwargs.get("rug", True)
        kwargs["rug"] = rug
        color = kwargs.get("color", "darkcyan")
        kwargs["color"] = color
        kde_kws = kwargs.get("kde_kws", {})
        cut = kde_kws.get("cut", 0)
        kde_kws["cut"] = cut
        kwargs["kde_kws"] = kde_kws
        ax = sns.distplot(outlier_scores, **kwargs)
        ax.set_xlabel("Outlier score")
        if self.outlier_quantile is not None:
            x = np.quantile(outlier_scores, self.outlier_quantile)
            ax.axvline(
                x,
                color=color,
                linestyle="--",
            )
            if len(ax.get_yticks()):
                y = ax.get_yticks()[-1] / 2
                ax.text(
                    x + 0.01,
                    y,
                    f"quantile: {self.outlier_quantile:.4f}\nvalue: {x:.4f}",
                    color=color,
                    verticalalignment="center",
                )

        return ax
