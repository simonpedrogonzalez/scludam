import os
import sys


import copy
import math
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Type, Union

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.coordinates import Distance, SkyCoord
from astropy.stats import biweight_location, biweight_scale, mad_std
from astropy.stats.sigma_clipping import sigma_clipped_stats
from attrs import define, field, validators, Factory
from bayes_opt import BayesianOptimization
from hdbscan import HDBSCAN, all_points_membership_vectors
from hdbscan.validity import validity_index
from KDEpy import FFTKDE
from scipy import ndimage, stats
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from skimage.feature import peak_local_max
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import RobustScaler
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.robust.scale import huber, hubers_scale
from typing_extensions import TypedDict

sys.path.append(os.path.join(os.path.dirname("opencluster"), "."))
from opencluster.detection2 import CountPeakDetector, DetectionResult
from opencluster.hkde import HKDE
from opencluster.masker import RangeMasker
from opencluster.membership3 import DBME
from opencluster.synthetic import three_clusters_sample, sample3c
from opencluster.utils import Colnames
from astropy.table.table import Table
from opencluster.stat_tests import (
    StatTest,
    RipleysKTest,
    DipDistTest,
    HopkinsTest,
    TestResult,
)
from opencluster.membership3 import DBME
from opencluster.shdbscan import SHDBSCAN

@define
class DEP:

    detector: CountPeakDetector
    det_cols: List[str]

    tests: List[StatTest]
    test_cols: List[List[str]] = field()

    membership_cols: List[str]
    clusterer: SHDBSCAN = SHDBSCAN(auto_allow_single_cluster=True, min_cluster_size=50)
    estimator: DBME = DBME()
    test_mode: str = field(
        default="any", validator=validators.in_(["any", "all", "majority"])
    )

    sample_sigma_factor: int = 5
    det_kws: dict = Factory(dict)

    colnames: Colnames = None

    test_results: List[List[TestResult]] = Factory(list)
    detection_result: DetectionResult
    proba: np.ndarray = None
    limits = Factory(list)
    masks = Factory(list)
    clusterers = Factory(list)
    estimators = Factory(list)
    is_clusterable = Factory(list)

    @test_cols.validator
    def test_cols_validator(self, attr, value):
        if len(value) != len(self.tests):
            raise ValueError("test_cols must have the same length as tests")

    def _check_cols(self, cols):
        if len(self.colnames.get_data_names(cols)) != len(cols):
            raise ValueError(
                "Columns must be a subset of {}".format(self.colnames.get_data_names())
            )

    def _detect(self, df: pd.DataFrame):
        self._check_cols(self.det_cols)
        detection_data = df[self.det_cols].values()
        detection_result = self.detector.detect(detection_data, **self.det_kws)
        self.detection_result = detection_result
        return detection_result

    def _get_region_mask(self, df: pd.DataFrame, center: np.ndarray, sigma: np.ndarray):
        self._check_cols(self.det_cols)
        detection_data = df[self.det_cols].values()
        sample_limits = np.vstack(
            (
                center - sigma * self.sample_sigma_factor,
                center + sigma * self.sample_sigma_factor,
            )
        ).T
        self.limits.append(sample_limits)
        mask = RangeMasker(limits).mask(detection_data)
        self.masks.append(mask)
        return mask

    def _test(self, df: pd.DataFrame):
        test_cols = list(set([item for sublist in self.test_cols for item in sublist]))
        test_df = df[test_cols]
        results = []
        for i, stat_test in enumerate(self.tests):
            data = test_df[self.test_cols[i]].values()
            results.append(stat_test.test(t_data))
        self.test_results.append(results)
        return results

    def _is_sample_clusterable(self, test_results: List[TestResult]):
        if len(test_results) == 0:
            is_clusterable = True
        trs = np.asarray([tr.passed for tr in test_results])
        if self.test_mode == "any":
            is_clusterable = np.any(trs)
        if self.test_mode == "all":
            is_clusterable = np.all(trs)
        if self.test_mode == "majority":
            is_clusterable = np.sum(trs) >= trs.size / 2
        else:
            raise ValueError("test_mode must be one of 'any', 'all', 'majority'")
        self.is_clusterable.append(is_clusterable)
        return is_clusterable

    def _estimate_membership(self, df: pd.DataFrame):

        # get err and corr columns and use them if they exist
        err_cols, missing_err = self.colnames.get_error_names(self.membership_cols)
        corr_cols, missing_corr = self.colnames.get_corr_names(self.membership_cols)

        if not missing_err:
            self.membership_cols += err_cols
            n_errs = len(err_cols)
        else:
            err = None
        if not missing_corr:
            self.membership_cols += corr_cols
            n_corrs = len(corr_cols)
        else:
            corr = None

        self._check_cols(self.membership_cols)

        # data to use for membership estimation
        data = df[self.membership_cols].values()

        # data to calculate center
        data_only = data[:, : -n_errs - n_corrs].values()
        center = np.array(
            [
                sigma_clipped_stats(
                    data_only[:, i],
                    cenfunc="median",
                    stdfunc="mad_std",
                    maxiters=None,
                    sigma=1,
                )
                for i in range(data_only.shape[1])
            ]
        )[:, 1]

        # create a clusterer for the data
        clusterer = copy.deepcopy(self.clusterer)
        if clusterer.min_cluster_size and clusterer.clusterer is None:
            clusterer.min_cluster_size = int(peak.count)

        clusterer.fit(data=data, centers=[center])
        self.clusterers.append(clusterer)

        init_proba = clusterer.proba

        estimator = copy.deepcopy(self.estimator)
        estimator.fit(data=data, init_proba=init_proba, err=err, corr=corr)
        self.estimators.append(estimator)

        return estimator.posteriors

    def fit(self, df):
        df = df.dropna()
        n, d = df.shape

        self.colnames = Colnames(df.columns)

        detection_result = self._detect(df)

        if not detection_result.centers.size:
            return np.ones(n).reshape(-1, 1)

        global_proba = []

        for i, peak_center in enumerate(detection_result.centers):

            mask = self._get_region_mask(df, peak_center)
            region_df = df[mask]

            test_results = self._test(region_df)

            if self._is_sample_clusterable(test_results):
                sample_proba = self._estimate_membership(df, mask)

                n_classes = sample_proba.shape[1]
                n_clusters = n_classes - 1

                # add each found cluster probs
                for n_c in range(n_clusters):
                    cluster_proba = np.zeros(n)
                    cluster_proba[mask] = proba[:, n_c + 1]
                    global_proba.append(cluster_proba)

        # add row for field prob
        global_proba = np.array(global_proba).T
        _, total_clusters = global_proba.shape
        result = np.empty((n, total_clusters + 1))
        result[:, 1:] = global_proba
        result[:, 0] = 1 - global_proba.sum(axis=1)
        self.proba = result

        return self

# def test_PMPlxPipeline():
#     df = sample3c()

#     p = Pipeline(
#         detector=CountPeakDetector(min_dif=50, bin_shape=[1, 1, 0.1]),
#         det_kws={"heatmaps": False},
#         det_cols=["pmra", "pmdec", "log10_parallax"],
#         tests=[
#             RipleysKTest(mode="chiu", pvalue_threshold=0.05),
#             RipleysKTest(mode="chiu", pvalue_threshold=0.05),
#         ],
#         test_cols=[["pmra", "pmdec"], ["ra", "dec"]],
#         membership_cols=["pmra", "pmdec", "parallax", "ra", "dec"],
#         sample_sigma_factor=3,
#     ).process(df)
#     _, n_clus = p.shape

#     for n in range(1):
#         sns.scatterplot(df.ra, df.dec, hue=p[:, n], hue_norm=(0, 1)).set(
#             title=f"p(x∈C{n}) ra-dec"
#         )
#         plt.figure()
#         sns.scatterplot(df.pmra, df.pmdec, hue=p[:, n], hue_norm=(0, 1)).set(
#             title=f"p(x∈C{n}) pm"
#         )
#         plt.figure()
#         sns.scatterplot(df.pmra, df.parallax, hue=p[:, n], hue_norm=(0, 1)).set(
#             title=f"p(x∈C{n}) pmra-plx"
#         )

#     plt.show()
#     print("coso")


def test_PMPlxPipeline_real_data():
    df = Table.read("tests/data/ngc2527_small.xml").to_pandas()

    dep = DEP(
        detector=CountPeakDetector(
            bin_shape=[0.5, 0.5, 0.05],
        ),
        det_cols=["pmra", "pmdec", "log10_parallax"],
        tests=[
            RipleysKTest(mode="chiu", pvalue_threshold=0.05),
        ],
        test_cols=[["pmra", "pmdec"]],
        membership_cols=["pmra", "pmdec", "parallax"],
        sample_sigma_factor=5,
    ).fit(df)
    print("coso")


# test_PMPlxPipeline()
test_PMPlxPipeline_real_data()
