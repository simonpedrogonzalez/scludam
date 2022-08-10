import os
import sys


import copy
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.coordinates import Distance, SkyCoord
from astropy.stats.sigma_clipping import sigma_clipped_stats
from attrs import define, field, validators, Factory
from hdbscan.validity import validity_index
from scipy import ndimage, stats
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import RobustScaler
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.robust.scale import huber, hubers_scale
from typing_extensions import TypedDict

sys.path.append(os.path.join(os.path.dirname("scludam"), "."))
from scludam.detection import CountPeakDetector, DetectionResult
from scludam.hkde import HKDE
from scludam.masker import RangeMasker
from scludam.membership import DBME
from scludam.utils import Colnames
from astropy.table.table import Table
from scludam.stat_tests import (
    StatTest,
    RipleysKTest,
    DipDistTest,
    HopkinsTest,
    TestResult,
)
from scludam.membership import DBME
from scludam.shdbscan import SHDBSCAN


@define
class DEP:

    detector: CountPeakDetector
    det_cols: List[str]

    tests: List[StatTest]
    test_cols: List[List[str]] = field()

    membership_cols: List[str]
    clusterer: SHDBSCAN = SHDBSCAN(
        auto_allow_single_cluster=True,
        min_cluster_size=50,
        noise_proba_mode="conservative",
        cluster_proba_mode="hard",
        scaler=RobustScaler(),
    )
    estimator: DBME = DBME()
    test_mode: str = field(
        default="any", validator=validators.in_(["any", "all", "majority"])
    )

    sample_sigma_factor: int = 1
    det_kws: dict = Factory(dict)

    colnames: Colnames = None

    test_results: List[List[TestResult]] = Factory(list)
    detection_result: DetectionResult = None
    proba: np.ndarray = None
    limits: List = Factory(list)
    masks: List = Factory(list)
    clusterers: List = Factory(list)
    estimators: List = Factory(list)
    is_clusterable: List = Factory(list)

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
        detection_data = df[self.det_cols].values
        detection_result = self.detector.detect(detection_data, **self.det_kws)
        self.detection_result = detection_result
        return detection_result

    def _get_region_mask(self, df: pd.DataFrame, center: np.ndarray, sigma: np.ndarray):
        detection_data = df[self.det_cols].values
        limits = np.vstack(
            (
                center - sigma * self.sample_sigma_factor,
                center + sigma * self.sample_sigma_factor,
            )
        ).T
        self.limits.append(limits)
        mask = RangeMasker(limits).mask(detection_data)
        self.masks.append(mask)
        return mask

    def _test(self, df: pd.DataFrame):
        test_cols = list(set([item for sublist in self.test_cols for item in sublist]))
        test_df = df[test_cols]
        results = []
        for i, stat_test in enumerate(self.tests):
            data = test_df[self.test_cols[i]].values
            results.append(stat_test.test(data))
        self.test_results.append(results)
        # from scludam.stat_tests import rripley
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # from sklearn.preprocessing import MinMaxScaler

        # dd = MinMaxScaler().fit_transform(data)
        # dd = dd[(dd[:, 0] > 0.25) & (dd[:, 0] < 0.75)]
        # dd = dd[(dd[:, 1] > 0.25) & (dd[:, 1] < 0.75)]
        # plt.figure()
        # sns.scatterplot(dd[:, 0], dd[:, 1])
        # plt.show()
        # plt.figure()
        # rr, rlf = rripley(data)
        # sns.lineplot(rr, rlf)
        # sns.lineplot(rr, rr)
        # sns.lineplot(results[0].radii, results[0].l_function)
        # plt.show()
        return results

    def _is_sample_clusterable(self, test_results: List[TestResult]):
        if len(test_results) == 0:
            is_clusterable = True
        trs = np.asarray([tr.rejectH0 for tr in test_results])
        if self.test_mode == "any":
            is_clusterable = np.any(trs)
        elif self.test_mode == "all":
            is_clusterable = np.all(trs)
        elif self.test_mode == "majority":
            is_clusterable = np.sum(trs) >= trs.size / 2
        else:
            raise ValueError("test_mode must be one of 'any', 'all', 'majority'")
        self.is_clusterable.append(is_clusterable)
        return is_clusterable

    def _estimate_membership(self, df: pd.DataFrame, count: int, center: np.ndarray):

        # get err and corr columns and use them if they exist
        err_cols, missing_err = self.colnames.get_error_names(self.membership_cols)
        corr_cols, missing_corr = self.colnames.get_corr_names(self.membership_cols)

        # data to calculate center
        data = df[self.membership_cols].values

        if not missing_err:
            err = df[err_cols].values
        else:
            err = None
        if not missing_corr:
            corr = df[corr_cols].values
        else:
            corr = None

        # create a clusterer for the data
        clusterer = copy.deepcopy(self.clusterer)
        if clusterer.min_cluster_size and clusterer.clusterer is None:
            clusterer.min_cluster_size = int(count)

        clusterer.fit(data=data, centers=[center])
        self.clusterers.append(clusterer)

        init_proba = clusterer.proba

        estimator = copy.deepcopy(self.estimator)
        estimator.fit(data=data, init_proba=init_proba, err=err, corr=corr)
        self.estimators.append(estimator)

        return estimator.posteriors

    # def _check_all_cols(self):

    def fit(self, df):
        df = df.dropna()
        n, d = df.shape

        self.colnames = Colnames(df.columns)

        self._check_cols(self.det_cols)
        self.detection_result = self._detect(df)

        if not self.detection_result.centers.size:
            return np.ones(n).reshape(-1, 1)

        self._check_cols(self.membership_cols)

        # global_proba = []

        for i, peak_center in enumerate(self.detection_result.centers):

            mask = self._get_region_mask(
                df, peak_center, self.detection_result.sigmas[i]
            )
            region_df = df[mask]

            test_results = self._test(region_df)

        #     if self._is_sample_clusterable(test_results):
        #         proba = self._estimate_membership(region_df, self.detection_result.counts[i], peak_center)

        #         n_classes = proba.shape[1]
        #         n_clusters = n_classes - 1

        #         # add each found cluster probs
        #         for n_c in range(n_clusters):
        #             cluster_proba = np.zeros(n)
        #             cluster_proba[mask] = proba[:, n_c + 1]
        #             global_proba.append(cluster_proba)

        # add row for field prob
        # global_proba = np.array(global_proba).T
        # _, total_clusters = global_proba.shape
        # result = np.empty((n, total_clusters + 1))
        # result[:, 1:] = global_proba
        # result[:, 0] = 1 - global_proba.sum(axis=1)
        # self.proba = result

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
    df = Table.read("/home/simon/test-scludam/data/ngc2527_data.xml").to_pandas()
    df["log10_parallax"] = np.log10(df["parallax"])
    dep = DEP(
        detector=CountPeakDetector(
            bin_shape=[0.5, 0.5, 0.05],
            max_n_peaks=10,
        ),
        det_cols=["pmra", "pmdec", "log10_parallax"],
        tests=[
            RipleysKTest(mode="ripley", pvalue_threshold=0.05, max_samples=1000),
            RipleysKTest(mode="chiu", pvalue_threshold=0.05, max_samples=1000),
            RipleysKTest(mode="ks", pvalue_threshold=0.05, max_samples=1000),
            # DipDistTest(pvalue_threshold=0.05, max_samples=1000),
            # HopkinsTest(pvalue_threshold=0.05, max_samples=1000),
        ],
        test_cols=[["pmra", "pmdec"]] * 3,
        membership_cols=["pmra", "pmdec", "log10_parallax"],
        sample_sigma_factor=5,
    ).fit(df)
    trs = dep.test_results
    for tt in trs:
        tr = tt[0]
        sns.lineplot(tr.radii, tr.radii)
        sns.lineplot(tr.radii, tr.l_function)
        plt.show()
        plt.figure()
    dep.detector.plot()
    plt.show()

    print("coso")


# test_PMPlxPipeline()
test_PMPlxPipeline_real_data()
