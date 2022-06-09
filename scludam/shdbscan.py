from scipy.stats import multivariate_normal

import os
import sys
from copy import deepcopy
from typing import Union

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from attrs import define, validators, field
from hdbscan import HDBSCAN, all_points_membership_vectors
from hdbscan.validity import validity_index
from sklearn.base import ClassifierMixin, ClusterMixin, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
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

sys.path.append(os.path.join(os.path.dirname("scludam"), "."))
from scludam.plots import (
    membership_plot,
    membership_3d_plot,
    probaplot,
    uniprobaplot,
    tsneplot,
)
from scludam.plot_gauss_err import plot_kernels
from scludam.utils import combinations, Colnames
from scludam.synthetic import BivariateUniform
from scludam.hkde import HKDE, PluginSelector, pair_density_plot
from scludam.masker import RangeMasker, DistanceMasker, CrustMasker
from scludam.utils import one_hot_encode


@define
class SHDBSCAN:
    clusterer: HDBSCAN = None
    min_cluster_size: int = field(default=None)
    allow_single_cluster: bool = False
    auto_allow_single_cluster: bool = False
    min_samples: int = None
    centers: list = None
    metric: str = "euclidean"
    data: np.ndarray = None
    n: int = None
    d: int = None
    n_classes: int = None
    proba: np.ndarray = None
    scaler: TransformerMixin = None  # RobustScaler()
    centers_provided: bool = False
    labels: np.ndarray = None
    unique_labels: np.ndarray = None
    clusters: bool = None
    noise_found: bool = None
    outlier_quantile: float = field(
        default=None,
        validator=validators.optional([validators.ge(0), validators.le(1)]),
    )

    @min_cluster_size.validator
    def min_cluster_size_validator(self, attribute, value):
        if value is None:
            if self.clusterer is None:
                raise ValueError(
                    "either min_cluster_size or clusterer must be provided"
                )
        elif value < 1:
            raise ValueError("min_cluster_size must be greater than 1")

    def cluster(self, data: np.ndarray):

        if self.centers is not None or self.auto_allow_single_cluster:
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
    def get_proba(self):

        """

        Outlier score is an implementation of GLOSH, it catches local
        outliers as well as just points that are far away from everything.
        Thus a point can be "in" a cluster, and have a label, but be sufficiently far
        from an otherwise very dense core that is is anomalous in that local region
        of space (i.e. it is weird to have a point there when almost everything else
        is far more tightly grouped).

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
        if not np.any(self.unique_labels == -1):
            one_hot_code = np.vstack(
                (np.zeros(one_hot_code.shape[0]), one_hot_code.T)
            ).T

        n, n_classes = one_hot_code.shape
        n_clusters = n_classes - 1

        outlier_scores = np.copy(self.clusterer.outlier_scores_)
        # sanitization just in case
        outlier_scores[outlier_scores == -np.inf] = 0
        outlier_scores[outlier_scores == np.inf] = 1

        if self.outlier_quantile is not None:
            # use quantile to determine scale outlier score
            # if we consider outlier for score > quantile
            # then we scale de outlier score with that threshold
            # as being the defining point when p(outlier) = 1
            top = np.quantile(outlier_scores, self.outlier_quantile)
            outlier_scores[outlier_scores > top] = top
            outlier_scores = (
                MinMaxScaler().fit_transform(outlier_scores.reshape(-1, 1)).ravel()
            )

        if self.clusterer.allow_single_cluster:
            noise_proba = np.vstack(
                (
                    one_hot_code[:, 0],
                    outlier_scores,
                    1 - self.clusterer.probabilities_,
                )
            ).max(0)
            cluster_proba = 1 - noise_proba
        else:
            membership = all_points_membership_vectors(self.clusterer)
            noise_proba = np.vstack(
                (
                    1 - membership.sum(axis=1),
                    one_hot_code[:, 0],
                    outlier_scores,
                    1 - self.clusterer.probabilities_,
                )
            ).max(0)

            cluster_proba = np.zeros((n, n_clusters))

            # way 1: get cl_proba from crude membership vector
            # problem: some points have high membership for different clusters (e.g. (c1: .3, cl2: .3, cl3: .3) )
            # those points, instead of being between clusters, are actually inside one of them
            # and make no sense. That significantly affects how the populations are estimated in the next phases
            """ full_cl_proba = (membership * membership.sum(axis=1, keepdims=True) / np.tile((1 - noise_proba), (self.n_classes-1, 1)).T)
            cluster_proba[1-noise_proba > 0] = full_cl_proba[1-noise_proba > 0] """

            # way 2: hard classify among clusters, so probs look like (noise: .8, cluster1: .2, cluster2: 0)
            # problem: when a point is really .5/.5 between 2 clusters, like in the middle point, it would be considererd
            # 1/0
            cluster_proba = (
                one_hot_code[:, 1:] * np.tile((1 - noise_proba), (n_clusters, 1)).T
            )

        if len(cluster_proba.shape) == 1:
            cluster_proba = np.atleast_2d(cluster_proba).T

        proba = np.zeros((n, n_classes))
        proba[:, 0] = noise_proba
        proba[:, 1:] = cluster_proba
        assert np.allclose(proba.sum(axis=1), 1)
        return proba

    def center_based_cluster_selection(self, data, labels, input_centers):

        # compares input cluster centers with obtained cluster centers
        # if input cluster centers are less than obtained, then select
        # onbtained clusters that match input centers the best

        cluster_labels = self.unique_labels[self.unique_labels != -1]
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
                    for i in range(self.d)
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

        posteriors = self.get_proba()

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
        new_posteriors = np.zeros((self.n, new_n_classes))
        new_posteriors[:, 0] = noise_proba
        new_posteriors[:, 1:] = cluster_proba

        assert np.allclose(new_posteriors.sum(axis=1), 1)

        # reorder kept labels
        for i, label in enumerate(best_solution):
            new_labels[new_labels == label] = i

        return new_labels, new_posteriors

    def scale(self, data, centers):
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        if self.centers_provided:
            centers = self.scaler.transform(centers)
        return data, centers

    def fit(self, data: np.ndarray, centers: np.ndarray = None):

        self.data = np.atleast_2d(np.asarray(data))

        if centers is not None:
            self.centers = np.atleast_2d(np.asarray(centers))
            if self.centers.shape[0] <= 0:
                self.centers_provided = False
            else:
                if self.centers.shape[1] != self.data.shape[1]:
                    raise ValueError(
                        "Provided centers have different number of dimensions than data"
                    )
                else:
                    self.centers_provided = True
        else:
            self.centers_provided = False

        self.n, self.d = self.data.shape

        if self.scaler:
            self.data, self.centers = self.scale(self.data, self.centers)

        self.cluster(self.data)

        self.labels = self.clusterer.labels_
        self.unique_labels = np.sort(np.unique(self.labels))
        self.n_classes = self.unique_labels.shape[0]

        # case only noise or only 1 cluster with no noise
        if self.n_classes == 1:
            self.proba = one_hot_encode(self.labels)
            return self

        n_clusters = self.unique_labels[self.unique_labels != -1].shape[0]
        # case cluster selection required
        if (
            not self.clusterer.allow_single_cluster
            and self.centers_provided
            and self.centers.shape[0] < n_clusters
        ):
            self.labels, self.proba = self.center_based_cluster_selection(
                self.data, self.labels, self.centers
            )
        else:
            self.proba = self.get_proba()
            self.labels = np.argmax(self.proba, axis=1) - 1

        self.unique_labels = np.sort(np.unique(self.labels))
        self.n_classes = self.unique_labels.shape[0]

        if not np.any(self.unique_labels == -1):
            # noise label should always be present even if there is no noise
            # because probability column is included
            self.unique_labels = np.array([-1] + list(self.unique_labels))

        return self

    def validity_index(self, **kwargs):
        return validity_index(self.data, self.labels, **kwargs)

    def plot(self, **kwargs):
        return probaplot(data=self.data, proba=self.proba, labels=self.labels, **kwargs)

    def tsneplot(self, **kwargs):
        return tsneplot(data=self.data, proba=self.proba, **kwargs)

    def memplot(self, label=0, **kwargs):
        return membership_plot(
            data=self.data,
            posteriors=self.proba[:, label + 1],
            labels=self.labels,
            **kwargs
        )

    def plot3d(self, label=0, **kwargs):
        return membership_3d_plot(
            self.data, self.proba[:, label + 1], self.labels + 1, **kwargs
        )

    def plot_out_dist(self, **kwargs):
        outlier_scores = self.clusterer.outlier_scores_[
            np.isfinite(self.clusterer.outlier_scores_)
        ]
        ax = sns.distplot(outlier_scores, rug=True, kde_kws={"cut": 0}, **kwargs)
        return ax


""" 
def iris():
    from sklearn.datasets import load_iris

    return load_iris().data


def plot3d(data, z, c):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter3D(data[:, 0], data[:, 1], z, c=c)
    return ax


def plot3d_s(data, z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_trisurf(
        data[:, 0], data[:, 1], z, cmap="viridis", linewidth=0, antialiased=False
    )
    return ax


def uniform_sample():
    return BivariateUniform(locs=(0, 0), scales=(1, 1)).rvs(1000)


def one_cluster_sample():
    sample = BivariateUniform(locs=(0, 0), scales=(1, 1)).rvs(900)
    sample2 = multivariate_normal(mean=(0.5, 0.5), cov=1.0 / 200).rvs(100)
    return np.concatenate((sample, sample2))


def two_clusters_sample():
    sample = BivariateUniform(locs=(0, 0), scales=(1, 1)).rvs(800)
    sample2 = multivariate_normal(mean=(0.75, 0.75), cov=1.0 / 200).rvs(100)
    sample3 = multivariate_normal(mean=(0.25, 0.25), cov=1.0 / 200).rvs(100)
    return np.concatenate((sample, sample2, sample3))


def test_0c():
    # no cluster should be found
    data = uniform_sample()
    s = SHDBSCAN(min_cluster_size=100).fit(data)
    s.plot()
    plt.show()
    print("coso")


def test_0c_force():
    # cluster is found because it is forced
    data = uniform_sample()
    s = SHDBSCAN(min_cluster_size=100, auto_allow_single_cluster=True).fit(
        data
    )  # or allow_single_cluster=True
    s.plot()
    plt.show()
    print("coso")


def test_1c():
    # no cluster is found
    data = one_cluster_sample()
    s = SHDBSCAN(min_cluster_size=100).fit(data)
    s.plot()
    plt.show()
    print("coso")


def test_1c_force():
    # cluster is found because it is forced
    data = one_cluster_sample()
    s = SHDBSCAN(
        min_cluster_size=100, auto_allow_single_cluster=True, outlier_quantile=0.9
    ).fit(data)
    s.plot()
    plt.show()
    all_points_membership_vectors(s.clusterer)
    print("coso")


def test_2c():
    # asc founds 1 cluster
    # not asc founds 2 clusters
    # aasc founds 2 clusters
    data = two_clusters_sample()
    s = SHDBSCAN(
        min_cluster_size=100, auto_allow_single_cluster=True, outlier_quantile=0.9
    ).fit(data)
    validity_index(s.data, s.labels)
    s.plot()
    plt.show()
    print("coso")


def test_2c_center():
    # finds correct cluster
    data = two_clusters_sample()
    s = SHDBSCAN(
        min_cluster_size=100, auto_allow_single_cluster=True, outlier_quantile=0.9
    ).fit(data, centers=np.array(((0.25, 0.25))))
    s.plot()
    plt.show()
    print("coso")


def test_iris():
    data = iris()
    s = SHDBSCAN(min_cluster_size=20, auto_allow_single_cluster=True).fit(data)
    s.tsneplot()
    plt.show()
    print("coso")


# test_iris()


def test_no_clusters_aasc(uniform_sample):
    shdbscan = SHDBSCAN(min_cluster_size=100, auto_allow_single_cluster=True).fit(
        uniform_sample
    )
    assert shdbscan.n_classes == 2
    assert shdbscan.proba.shape == (uniform_sample.shape[0], 2)


# test_no_clusters_aasc(uniform_sample())


def three_clusters_sample():
    sample = BivariateUniform(locs=(0, 0), scales=(1, 1)).rvs(500)
    sample2 = multivariate_normal(mean=(0.75, 0.75), cov=1.0 / 200).rvs(160)
    sample3 = multivariate_normal(mean=(0.25, 0.25), cov=1.0 / 200).rvs(160)
    sample4 = multivariate_normal(mean=(0.5, 0.5), cov=1.0 / 200).rvs(160)
    return np.concatenate((sample, sample2, sample3, sample4))


def test_three_clusters_two_centers(three_clusters_sample):
    shdbscan = SHDBSCAN(min_cluster_size=80, auto_allow_single_cluster=True).fit(
        three_clusters_sample, centers=[(0.75, 0.75), (0.5, 0.5)]
    )
    assert shdbscan.n_classes == 3
    assert shdbscan.proba.shape == (three_clusters_sample.shape[0], 3)
    # found correct one and cluster order preserved
    center = three_clusters_sample[shdbscan.labels == 0].mean(axis=0)
    assert center - np.array([0.25, 0.25]) < center - np.array([0.75, 0.75])
    center2 = three_clusters_sample[shdbscan.labels == 1].mean(axis=0)
    assert center2 - np.array([0.75, 0.75]) < center2 - np.array([0.25, 0.25])


# test_three_clusters_two_centers(three_clusters_sample())


def test_clusterer_parameter(iris):
    shdbscan = SHDBSCAN(
        clusterer=HDBSCAN(
            min_cluster_size=30,
            min_samples=10,
            allow_single_cluster=True,
        ),
        auto_allow_single_cluster=True,
    ).fit(iris)
    assert shdbscan.n_classes == 3
    assert shdbscan.proba.shape == (iris.shape[0], 3)
    assert shdbscan.clusterer.min_cluster_size == 30
    assert shdbscan.clusterer.min_samples == 10
    assert shdbscan.clusterer.allow_single_cluster == False


# test_clusterer_parameter(iris())
 """
