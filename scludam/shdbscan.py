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


"""Module for clustering of numerical data based on the HDBSCAN method.

This module provides a wrapper class for HDBSCAN that adds some extra functionality
regarding:
*  Calculation of probability like scores from the soft clustering method.
*  Cluster selection based on geometric centers.
*  Custom plots to visualize the results.

References
----------
.. [1]  HDBSCAN: Hierarchical density-based spatial clustering of applications with noise.

"""

from itertools import permutations
from typing import Callable, Optional, Union

import numpy as np
import seaborn as sns
from astropy.stats.sigma_clipping import sigma_clipped_stats
from attrs import define, field, validators
from hdbscan import HDBSCAN, all_points_membership_vectors
from hdbscan.validity import validity_index
from sklearn.base import TransformerMixin
from sklearn.metrics import pairwise_distances

from scludam.type_utils import (
    Numeric1DArray,
    Numeric2DArray,
    Numeric2DArrayLike,
    OptionalNumeric2DArrayLike,
    _type,
)

from scludam.plots import (
    pairprobaplot,
    scatter3dprobaplot,
    surfprobaplot,
    tsneprobaplot,
)
from scludam.utils import one_hot_encode


@define
class SHDBSCAN:
    # input attrs
    clusterer: HDBSCAN = None
    min_cluster_size: Optional[int] = field(
        default=None, validator=_type(Optional[int])
    )
    allow_single_cluster: bool = field(default=False, validator=_type(bool))
    auto_allow_single_cluster: bool = field(default=False, validator=_type(bool))
    min_samples: Optional[int] = field(default=None, validator=_type(Optional[int]))
    metric: Union[str, Callable] = field(default="euclidean")
    outlier_quantile: Optional[float] = field(
        default=None,
        validator=[
            _type(Optional[float]),
            validators.optional([validators.ge(0), validators.le(1)]),
        ],
    )
    noise_proba_mode: str = field(
        default="score",
        validator=[_type(str), validators.in_(["score", "outlier", "conservative"])],
    )
    cluster_proba_mode: str = field(
        default="soft", validator=[_type(str), validators.in_(["soft", "hard"])]
    )
    scaler: Optional[TransformerMixin] = field(
        default=None, validator=_type(Optional[TransformerMixin])
    )  # RobustScaler()

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

        """Outlier score is an implementation of GLOSH, it catches local outliers as
        well as just points that are far away from everything. Thus a point can be "in"
        a cluster, and have a label, but be sufficiently far from an otherwise very
        dense core that is is anomalous in that local region of space (i.e. it is weird
        to have a point there when almost everything else is far more tightly grouped).

        The probabilties are slightly misnamed. It is essentially a "cluster membership score"
        that is, if the point is in a cluster how relatively well tied to the cluster is it?
        It is effectively the ratio of the distance scale at which this point fell out of the cluster
        with the distance scale of the core of the cluster.

        Any noise points are assigned a probability 0 as it is the "membership score"
        for the cluster that they are a member of, and noise points are not a member of any cluster.

        https://github.com/scikit-learn-contrib/hdbscan/issues/80

        """
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

    def fit(self, data: Numeric2DArray, centers: OptionalNumeric2DArrayLike = None):

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

    def validity_index(self, **kwargs):
        return validity_index(self._data, self.labels, **kwargs)

    def pairplot(self, **kwargs):
        return pairprobaplot(
            data=self._data, proba=self.proba, labels=self.labels, **kwargs
        )

    def tsneplot(self, **kwargs):
        return tsneprobaplot(
            data=self._data, proba=self.proba, labels=self.labels, **kwargs
        )

    def scatter3dplot(self, **kwargs):
        return scatter3dprobaplot(self._data, self.proba, **kwargs)

    def surfplot(self, **kwargs):
        return surfprobaplot(self._data, self.proba, **kwargs)

    def outlierplot(self, **kwargs):
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
            y = ax.get_yticks()[-1] / 2
            ax.text(
                x + 0.01,
                y,
                f"quantile: {self.outlier_quantile:.4f}\nvalue: {x:.4f}",
                color=color,
                verticalalignment="center",
            )

        return ax
