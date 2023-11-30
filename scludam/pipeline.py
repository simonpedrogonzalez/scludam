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


"""Module with default Detection and Membership Estimation Pipeline.

All steps of the process can be configured following the documentation
for each function.

Examples
--------
.. literalinclude:: ../../examples/pipeline/dep.py
    :language: python
    :linenos:

.. image:: ../../examples/pipeline/dep_pmra_pmdec.png

"""

import copy
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from astropy.table.table import Table
from attrs import Factory, define, field, validators
from beartype import beartype
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from scludam.detection import CountPeakDetector, DetectionResult
from scludam.fetcher import search_objects_near_data, simbad2gaiacolnames
from scludam.masker import RangeMasker
from scludam.membership import DBME
from scludam.plots import plot_objects, scatter2dprobaplot
from scludam.shdbscan import SHDBSCAN
from scludam.stat_tests import StatTest, TestResult
from scludam.utils import Colnames


@define
class DEP:
    """Detection and membership estimation pipeline.

    Class for running detection and membership estimation
    in a dataframe.

    Attributes
    ----------
    detector : CountPeakDetector
        Detector to use, it is required and should not be fitted.
    det_cols : List[str]
        Columns to use for detection. Should be a subset of
        the columns in the dataframe to be used.
    sample_sigma_factor : float, optional
        Factor to multiply the sigma of the detection region, in order
        to get a region sample for the overdensity detected, by default 1.
        Take into account that the sigma used is currently the
        bin shape in each dimension.
    tests : List[StatTest], optional
        Statistical tests to use for detection, optional, by default [].
        The list can include a non fitted
        instance of :class:`~scludam.stat_tests.StatTest`.
    test_cols : List[List[str]], optional
        Columns to use for statistical tests, optional, by default [].
        Note that the list must have the same length as the tests list,
        and each item is a list of columns to use for the statistical test.
    test_mode : str, optional
        Mode to use for statistical tests, optional, by default 'any'. If 'any',
        the test is considered passed if any of the stat tests results
        in the rejection of their null hypothesis. Other options
        are 'all' and 'majority'.
    clusterer : SHDBSCAN, optional
        Clusterer to be used to get the initial probabilities. By default,
        an instance of :class:`~scludam.shdbscan.SHDBSCAN` with the
        following parameters:
        auto_allow_single_cluster=True, noise_proba_mode="conservative",
        cluster_proba_mode="hard", scaler=RobustScaler()
        (from sklearn.preprocessing).
    estimator : DBME, optional
        Estimator to use for membership estimation, by default an instance of
        :class:`~scludam.membership.DBME` with the default parameters.
    mem_cols : List[str], optional
        Columns to use for membership estimation, by default, ``det_cols``.
        Should be a subset of the columns in the dataframe to be used. It is
        recommended to use the same columns as for detection, or at least some
        of them, so center estimation can be used for better clustering. For the
        estimation process, error and correlation columns can be used if they are
        available in the dataframe. For example, if mem_cols is ['x', 'y'],
        the program will check if columns 'x_error', 'y_error', 'x_y_corr' are
        present.
    n_detected : int
        Output attribute, number of overdensities detected.
    detection_result: DetectionResult
        Output attribute, result of the detection.
    test_results: List[TestResult]
        Output attribute, list of statistical test results.
    is_clusterable: List[bool]
        Output attribute, list of decisions taken
        in the tests, whether the overdensity can be clustered.
    n_estimated : int
        Output attribute, number of overdensities estimated, that is,
        overdensities that passed the tests.
    proba : np.ndarray
        Output attribute, membership probabilities of each
        cluster found.
    labels : np.ndarray
        Output attribute, label of each data point, starting
        from -1 (noise).
    limits: List[Numeric1DArray]
        Output attribute, list of limits used for each detection region.
    masks: List[NDArray[bool]]
        Output attribute, list of masks used for each detection region.
    clusterers: List[SHDBSCAN]
        Output attribute, list of clusterers used for each detection region.
    estimators: List[DBME]
        Output attribute, list of estimators used for each detection region.

    """

    # input attributes
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
    sample_sigma_factor: int = 1

    # output attributes
    n_detected: int = None
    n_estimated: int = None
    test_results: List[List[TestResult]] = Factory(list)
    detection_result: DetectionResult = None
    proba: np.ndarray = None
    labels: np.ndarray = None
    limits: List = Factory(list)
    masks: List = Factory(list)
    clusterers: List = Factory(list)
    estimators: List = Factory(list)
    is_clusterable: List = Factory(list)

    # internal attributes
    _df: pd.DataFrame = None
    _colnames: Colnames = None
    _objects: pd.DataFrame = None

    @test_cols.validator
    def _test_cols_validator(self, attr, value):
        if len(value) != len(self.tests):
            raise ValueError("test_cols must have the same length as tests")

    def __attrs_post_init__(self):
        """Attrs initialization.

        Do not execute.

        """
        if self.mem_cols is None:
            self.mem_cols = self.det_cols
        elif sorted(self.mem_cols) == sorted(self.det_cols):
            self.mem_cols = sorted(self.det_cols)

    def _check_cols(self, cols):
        if len(self._colnames.data(cols)) != len(cols):
            raise ValueError(
                "Columns must be a subset of {}".format(self._colnames.data())
            )

    def _detect(self, df: pd.DataFrame):
        detection_data = df[self.det_cols].values
        detection_result = self.detector.detect(detection_data)
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
        err_cols = self._colnames.error(self.mem_cols)
        corr_cols = self._colnames.corr(self.mem_cols)

        if not self._colnames.missing_error(self.mem_cols):
            err = df[err_cols].values
        else:
            err = None
        if not self._colnames.missing_corr(self.mem_cols):
            corr = df[corr_cols].values
        else:
            corr = None

        # estimate membershipts
        estimator = copy.deepcopy(self.estimator)
        estimator.fit(data=data, init_proba=clusterer.proba, err=err, corr=corr)
        self.estimators.append(estimator)

        return estimator.posteriors

    @beartype
    def fit(self, df: pd.DataFrame):
        """Perform the detection and membership estimation.

        NaNs are dropped from the dataframe copy and are
        not taken into account.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame.

        Returns
        -------
        DEP
            Fitted instance of DEP.

        """
        df = df.dropna()

        self._df = df

        n, d = df.shape

        # check all columns
        self._colnames = Colnames(df.columns)
        self._check_cols(self.det_cols)
        self._check_cols(self.mem_cols)
        for i in range(len(self.test_cols)):
            self._check_cols(self.test_cols[i])

        # detect
        print("detecting overdensities...")
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
        print(f"found {self.detection_result.centers.shape[0]} overdensities")
        for i, center in enumerate(self.detection_result.centers):
            count = self.detection_result.counts[i]
            sigma = self.detection_result.sigmas[i]
            mask = self._get_region_mask(df, center, sigma)
            region_df = df[mask]

            # test
            print(f"testing peak {i}...")
            test_results = self._test(region_df)

            if self._is_sample_clusterable(test_results):
                print(f"estimating membership of peak {i}...")
                proba = self._estimate_membership(region_df, count, center)

                n_classes = proba.shape[1]
                n_clusters = n_classes - 1

                # if np.any(proba) > 1 or np.any(proba) < 0:
                #     print("stop")

                # add each found cluster probs
                for n_c in range(n_clusters):
                    cluster_proba = np.zeros(n)
                    cluster_proba[mask] = proba[:, n_c + 1]
                    global_proba.append(cluster_proba)

        # add row for field prob
        global_proba = np.array(global_proba).T
        if global_proba.size == 0:
            self.n_detected = 0
            self.n_estimated = 0
            self.proba = np.ones(n).reshape(-1, 1)
            return self
        _, total_clusters = global_proba.shape
        col_idx = global_proba.argmax(axis=1) + 1
        row_idx = np.arange(0, global_proba.shape[0])
        idx = tuple(map(tuple, np.stack([row_idx, col_idx])))
        result = np.zeros((n, total_clusters + 1))
        # in tcase of region overlap, only the highest prob is kept

        result[idx] = global_proba.max(axis=1)
        result[:, 0] = 1 - result[idx]
        self.proba = result
        self.labels = np.argmax(self.proba, axis=1) - 1
        self.n_detected = self.detection_result.centers.shape[0]
        self.n_estimated = self.proba.shape[1] - 1

        return self

    def _is_fitted(self):
        return self.proba is not None

    @beartype
    def proba_df(self):
        """Return the data frame with the probabilities.

        Returns the full dataframe used for the process
        added columns for the labels and the probabilites.

        Returns
        -------
        pd.DataFrame
            Data with probabilities

        Raises
        ------
        Exception
            If DEP instance is not fitted.

        """
        if not self._is_fitted():
            raise Exception("Not fitted, try running fit()")

        cols = [f"proba({i-1})" for i in range(self.proba.shape[1])]
        df = pd.DataFrame(self.proba, columns=cols)
        df["label"] = self.labels
        return pd.concat(
            [self._df.reset_index(drop=True), df.reset_index(drop=True)],
            axis=1,
            sort=False,
        )

    @beartype
    def write(self, path: str, **kwargs):
        """Write the data frame with the probabilities to a file.

        Writes the data frame used for the process
        with labels and probabilities to a file. kwargs are
        passed to the astropy.table.Table.write method.
        Default kwargs are "overwrite"=True and
        "format"="fits".

        Parameters
        ----------
        path : str
            Full filepath with filename.

        Raises
        ------
        Exception
            If DEP instance is not fitted.

        """
        if not self._is_fitted():
            raise Exception("Not fitted, try running fit()")
        df = self.proba_df()
        table = Table.from_pandas(df)
        default_kws = {
            "overwrite": True,
            "format": "fits",
        }
        default_kws.update(kwargs)
        return table.write(path, **default_kws)

    @beartype
    def cm_diagram(
        self,
        cols: str = ["bp_rp", "phot_g_mean_mag"],
        plotcols: Optional[List[str]] = None,
        plot_objects: bool = True,
        **kwargs,
    ):
        """Color-magnitude diagram.

        Plots a 2d color magnitude diagram of the
        data, labels and probabilities. kwargs are passed to the
        :func:`~scludam.plots.scatter2dprobaplot`,
        some useful kwargs are "select_labels" for choosing which
        clusters to plot and "palette" for choosing the color palette.

        Parameters
        ----------
        cols : list, optional
            Dataframe columns to be used,
            by default ["bp_rp", "phot_g_mean_mag"].
            If the columns are not present in the dataframe, a KeyError
            is raised.
        plotcols : List[str], optional
            Colnames used for the axis labels in the plot, by default None.
        plot_objects : bool, optional
            Whether to plot objects found in the data region, by default True.
            By default the objects retrieved are of simbad otype "Cl*", meaning
            star clusters. This can be changed executing the function
            :func:`~scludam.pipeline.DEP.get_simbad_objects`.

        Returns
        -------
        Axes
            Axis of the plot

        Raises
        ------
        Exception
            If DEP instance is not fitted.

        """
        if not self._is_fitted():
            raise Exception("Not fitted, try running fit()")
        df = self._df[cols]
        ax = scatter2dprobaplot(df, self.proba, self.labels, plotcols, **kwargs)
        ax.invert_yaxis()
        if plot_objects:
            self._plot_objects(ax, cols)
        return ax

    @beartype
    def radec_plot(
        self,
        cols: str = ["ra", "dec"],
        plotcols: Optional[List[str]] = None,
        plot_objects: bool = True,
        **kwargs,
    ):
        """Color-magnitude diagram.

        Plots a 2d color magnitude diagram of the
        data, labels and probabilities. kwargs are passed to the
        :func:`~scludam.plots.scatter2dprobaplot`,
        some useful kwargs are "select_labels" for choosing which
        clusters to plot and "palette" for choosing the color palette.


        Parameters
        ----------
        cols : list, optional
            Dataframe columns to be used,
            by default ["bp_rp", "phot_g_mean_mag"].
            If the columns are not present in the dataframe, a KeyError
            is raised.
        plotcols : List[str], optional
            Colnames used for the axis labels in the plot, by default None.
        plot_objects : bool, optional
            Whether to plot objects found in the data region, by default True.
            By default the objects retrieved are of simbad otype "Cl*", meaning
            star clusters. This can be changed executing the function
            :func:`~scludam.pipeline.DEP.get_simbad_objects`.

        Returns
        -------
        Axes
            Axis of the plot

        Raises
        ------
        Exception
            If DEP instance is not fitted.

        """
        if not self._is_fitted():
            raise Exception("Not fitted, try running fit()")
        df = self._df[cols]
        ax = scatter2dprobaplot(df, self.proba, self.labels, plotcols, **kwargs)
        ax.invert_xaxis()
        if plot_objects:
            self._plot_objects(ax, cols)
        return ax

    @beartype
    def scatterplot(
        self,
        cols: List[str] = ["pmra", "pmdec"],
        plotcols: Optional[List[str]] = None,
        plot_objects: bool = True,
        **kwargs,
    ):
        """Scatter plot with results.

        Plots a 2d scatterplot of the data, labels and probabilities.
        kwargs are passed to the :func:`~scludam.plots.scatter2dprobaplot`,
        some useful kwargs are "select_labels" for choosing which
        clusters to plot and "palette" for choosing the color palette.

        Parameters
        ----------
        cols : List[str], optional
            Dataframe columns to be used, by default ["ra", "dec"].
            If the columns are not present in the dataframe, a KeyError
            is raised.
        plotcols : Optional[List[str]], optional
            Names of the axes labels to be used, by default None. If
            None, ``cols`` are used.
        plot_objects : bool, optional
            Whether to plot simbad objects found in the data region,
            by default True.
            By default the objects retrieved are of simbad otype "Cl*", meaning
            star clusters. This can be changed executing the function
            :func:`~scludam.pipeline.DEP.get_simbad_objects`.

        Returns
        -------
        Axes
            Axis of the plot

        Raises
        ------
        Exception
            If DEP instance is not fitted.

        """
        if not self._is_fitted():
            raise Exception("Not fitted, try running fit()")
        df = self._df[cols]
        ax = scatter2dprobaplot(df, self.proba, self.labels, plotcols, **kwargs)
        if plot_objects:
            self._plot_objects(ax, cols)
        return ax

    def _plot_objects(self, ax, cols, otype="Cl*"):
        try:
            if self._objects is None:
                self._objects = self.get_simbad_objects(otype=otype)
            objects_to_plot = simbad2gaiacolnames(self._objects).to_pandas()
            plot_objects(objects_to_plot, ax, cols)
            return ax
        except Exception as e:
            warnings.warn(f"Could not plot objects: {e}")

    def get_simbad_objects(self, **kwargs):
        """Get simbad objects found in the data region.

        kwargs are passed to the
        :func:`~scludam.fetcher.search_objects_near_data`
        function. If executed, scatterplot and cm_diagram will
        plot the objects found in the data region.

        Returns
        -------
        astropy.table.Table
            Table with the objects found.

        Raises
        ------
        Exception
            If DEP instance is not fitted.

        """
        print("getting simbad objects...")
        if not self._is_fitted():
            raise Exception("Not fitted, try running fit()")
        self._objects = search_objects_near_data(self._df, **kwargs)
        return self._objects
