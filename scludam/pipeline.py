import copy
from typing import List

import numpy as np
import pandas as pd
from astropy.table.table import Table
from attrs import Factory, define, field, validators
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from scludam.detection import CountPeakDetector, DetectionResult
from scludam.masker import RangeMasker
from scludam.membership import DBME
from scludam.plots import cm_diagram, scatter2dprobaplot
from scludam.shdbscan import SHDBSCAN
from scludam.stat_tests import StatTest, TestResult
from scludam.utils import Colnames


@define
class DEP:

    detector: CountPeakDetector
    det_cols: List[str]

    tests: List[StatTest] = Factory(list)
    test_cols: List[List[str]] = field(factory=list)
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
    mem_cols: List[str] = field(default=None)

    n_detected: int = None
    n_estimated: int = None
    test_range: float = 0.5
    sample_sigma_factor: int = 1
    det_kws: dict = Factory(dict)

    colnames: Colnames = None

    test_results: List[List[TestResult]] = Factory(list)
    detection_result: DetectionResult = None
    proba: np.ndarray = None
    labels: np.ndarray = None
    limits: List = Factory(list)
    masks: List = Factory(list)
    clusterers: List = Factory(list)
    estimators: List = Factory(list)
    is_clusterable: List = Factory(list)
    _df: pd.DataFrame = None

    @test_cols.validator
    def test_cols_validator(self, attr, value):
        if len(value) != len(self.tests):
            raise ValueError("test_cols must have the same length as tests")

    def __attrs_post_init__(self):
        if self.mem_cols is None:
            self.mem_cols = self.det_cols
        elif sorted(self.mem_cols) == sorted(self.det_cols):
            self.mem_cols = sorted(self.det_cols)

    def _check_cols(self, cols):
        if len(self.colnames.data(cols)) != len(cols):
            raise ValueError(
                "Columns must be a subset of {}".format(self.colnames.data())
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
        if len(self.tests) == 0:
            return None
        test_cols = list(set([item for sublist in self.test_cols for item in sublist]))
        test_df = df[test_cols]
        results = []

        for i, stat_test in enumerate(self.tests):
            data = test_df[self.test_cols[i]].values
            data = MinMaxScaler().fit_transform(data)
            # mask = RangeMasker(limits=[[.5-self.test_range, .5+self.test_range]]*data.shape[1]).mask(data)
            # data = data[mask]
            # sns.scatterplot(data[:, 0], data[:, 1])
            results.append(stat_test.test(data))

        self.test_results.append(results)
        return results

    def _is_sample_clusterable(self, test_results: List[TestResult]):
        if test_results is None:
            return True
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

        data = df[self.mem_cols].values

        # create a clusterer for the data
        # with the configuration defined by the user
        clusterer = copy.deepcopy(self.clusterer)
        if clusterer.min_cluster_size and clusterer.clusterer is None:
            clusterer.min_cluster_size = int(count)

        if set(self.mem_cols) <= set(self.det_cols):
            # if possible, use the center for
            # cluster selection
            if len(self.mem_cols) != len(self.det_cols):
                center = center.take([self.det_cols.index(c) for c in self.mem_cols])
            clusterer.fit(data=data, centers=[center])
        else:
            clusterer.fit(data=data)

        self.clusterers.append(clusterer)

        # get err and corr columns and use them if they exist
        err_cols = self.colnames.error(self.mem_cols)
        corr_cols = self.colnames.corr(self.mem_cols)

        if not self.colnames.missing_error(self.mem_cols):
            err = df[err_cols].values
        else:
            err = None
        if not self.colnames.missing_corr(self.mem_cols):
            corr = df[corr_cols].values
        else:
            corr = None

        # estimate membershipts
        estimator = copy.deepcopy(self.estimator)
        estimator.fit(data=data, init_proba=clusterer.proba, err=err, corr=corr)
        self.estimators.append(estimator)

        return estimator.posteriors

    def fit(self, df):
        df = df.dropna()

        self._df = df

        n, d = df.shape

        # check all columns
        self.colnames = Colnames(df.columns)
        self._check_cols(self.det_cols)
        self._check_cols(self.mem_cols)
        for i in range(len(self.test_cols)):
            self._check_cols(self.test_cols[i])

        print("detect")
        # detect
        self.detection_result = self._detect(df)

        # if no clusters detected, return full noise probs
        if not self.detection_result.centers.size:
            self.n_detected = 0
            self.n_estimated = 0
            self.proba = np.ones(n).reshape(-1, 1)
            return self

        global_proba = []

        # scatter_with_coors(df[["pmra", "pmdec"]], self.detection_result.centers)
        # plt.show()

        for i, center in enumerate(self.detection_result.centers):

            count = self.detection_result.counts[i]
            sigma = self.detection_result.sigmas[i]
            mask = self._get_region_mask(df, center, sigma)
            region_df = df[mask]

            # test
            print("test")
            test_results = self._test(region_df)

            if self._is_sample_clusterable(test_results):
                print("estimate")
                proba = self._estimate_membership(region_df, count, center)

                n_classes = proba.shape[1]
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

        self.labels = np.argmax(self.proba, axis=1) - 1

        self.n_detected = self.detection_result.centers.shape[0]
        self.n_estimated = self.proba.shape[1] - 1

        # labels = np.argmax(result, axis=1)-1
        # plt.show()
        # plotcols=[r"$\mu_\alpha cos(\delta) [mas / yr]$", r"$\mu_\delta [mas / yr]$", r"$\varpi [mas]$"]
        # cm_diag_plotcols=[r"$BP - RP [mag]$", r"$G [mag]$"]

        # cm_diagram(df[['bp_rp', 'phot_g_mean_mag']], self.proba, labels, cols=cm_diag_plotcols)

        # self.detector.plot(x=2, y=0, labels=plotcols, annot_prec=0)
        # self.estimators[0]._estimators[1].plot(100, cols=plotcols)
        # self.estimators[0].scatter3dplot(alpha=.9, cols=plotcols)
        # self.estimators[0].pairplot(diag_kws={'bins':20}, diag_kind="hist")

        # plt.show()
        # params = []
        # for i in range(self.proba.shape[1]-1):
        #     params.append(df[self.proba[:,i+1]>.5].mean())
        # from astropy.coordinates import SkyCoord
        # from scludam.fetcher import search_object
        # import astropy.units as u
        # pmra = search_object("ngc2527").table["PMRA"][0]
        # pmdec = search_object("ngc2527").table["PMDEC"][0]

        # sc = SkyCoord(ra=82.0312586022976*u.deg, dec=34.428328209365645*u.deg)
        # scatter_with_coors(
        #     df[['ra','dec']],
        #     coors=np.array([((121.24166667, -28.14666667), (params[0]['ra'], params[0]['dec'])]),
        #     hue=self.proba[:,1]
        # )
        # scatter_with_coors(
        #     df[['pmra','pmdec']],
        #     coors=np.array([ (pmra, pmdec),(params[0]['pmra'], params[0]['pmdec'])]),
        #     hue=self.proba[:,1]
        # )
        # print(params[i])
        # for i in range(10):
        #     region_df = pd.read_csv(f"reg{i}.csv")
        #     test_results = self._test(region_df)

        return self

    def _is_fitted(self):
        return self.proba is not None

    def probadf(self):
        if not self._is_fitted():
            raise Exception("Not fitted, try running fit()")

        cols = [f"proba({i-1})" for i in range(self.proba.shape[1])]
        df = pd.DataFrame(self.proba, columns=cols)
        df["label"] = self.labels
        return pd.concat([self._df, df], axis=1, sort=False)

    def write(self, path: str, **kwargs):
        if not self._is_fitted():
            raise Exception("Not fitted, try running fit()")
        df = self.probadf()
        table = Table.from_pandas(df)
        default_kws = {
            "overwrite": True,
            "format": "fits",
        }
        default_kws.update(kwargs)
        return table.write(path, **default_kws)

    def cm_diagram(self, cols=["bp_rp", "phot_g_mean_mag"], plotcols=None, **kwargs):
        if not self._is_fitted():
            raise Exception("Not fitted, try running fit()")
        df = self._df[cols]
        return cm_diagram(df, self.proba, self.labels, plotcols, **kwargs)

    def scatterplot(self, cols=["ra", "dec"], plotcols=None, **kwargs):
        if not self._is_fitted():
            raise Exception("Not fitted, try running fit()")
        df = self._df[cols]
        return scatter2dprobaplot(df, self.proba, self.labels, plotcols, **kwargs)


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


# def test_PMPlxPipeline_real_data():
#     # file = "/home/simon/test-scludam/data/ngc2527_data.xml"
#     # file = "/home/simon/test-scludam/data/stock_8_data.xml"
#     # file = "/home/simon/test-scludam/data/stock_8_data.fits"
#     base = "/home/simon/test-scludam/data"
#     files = [
#         "stock_8_data.fits",
#         "ngc2527_data.fits",
#         "ic2395_data.fits",
#         "ngc2323_data.fits",
#         "ic4665_data.fits",
#         "ic2602_data.fits",
#         "ngc2168_data.fits",
#     ]
#     file = f"{base}/{files[0]}"

#     df = Table.read(file).to_pandas()
#     df["log10_parallax"] = np.log10(df["parallax"])
#     print("read")
#     dep = DEP(
#         detector=CountPeakDetector(
#             bin_shape=[0.5, 0.5, 0.05],
#             max_n_peaks=2,
#             min_count=10,
#             min_score=3,
#         ),
#         estimator=DBME(
#             n_iters=2,
#         ),
#         det_cols=["pmra", "pmdec", "log10_parallax"],
#         mem_cols=["pmra", "pmdec"],  # , "log10_parallax", "ra", "dec"],
#         tests=[
#             RipleysKTest(pvalue_threshold=0.05, max_samples=100),  # factor=2.3,
#             HopkinsTest(),
#         ],
#         test_cols=[["ra", "dec"]] * 2,
#         sample_sigma_factor=2,
#     ).fit(df)
#     print("done")
#     dep.scatterplot()
#     # dep.write('ic2168dep.fits')
#     trs = dep.test_results
#     th0 = [[t.rejectH0 for t in tt] for tt in trs]
#     # for tt in trs:
#     #     tr = tt[0]
#     #     sns.lineplot(tr.radii, tr.radii)
#     #     sns.lineplot(tr.radii, tr.l_function)
#     #     plt.show()
#     #     plt.figure()
#     # dep.detector.plot()
#     # plt.show()
#     print("coso")


# # test_PMPlxPipeline()
# test_PMPlxPipeline_real_data()
