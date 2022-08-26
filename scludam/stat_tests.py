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

"""Module for useful statistical tests.

The tests in this module can be used to determine if there is cluster structure (the
data is "clusterable") in a n dimensional numerical dataset.

"""

from abc import abstractmethod
from copy import deepcopy
from numbers import Number
from typing import Union
from warnings import warn

import numpy as np
from astropy.stats import RipleysKEstimator
from attrs import define, field, validators
from beartype import beartype
from diptest import diptest
from scipy.stats import beta, ks_2samp
from sklearn.base import TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree, DistanceMetric
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from scludam.type_utils import Numeric1DArray, Numeric2DArray


@define
class TestResult:
    """Base class to hold the results of a statistical test.

    Attributes
    ----------
    rejectH0 : bool
        Whether the null hypothesis was rejected.

    """

    rejectH0: bool


class StatTest:
    """Base class for statistical tests."""

    @abstractmethod
    def test(self, data: Numeric2DArray, *args, **kwargs) -> TestResult:
        """Perform the test.

        Parameters
        ----------
        data : Numeric2DArray
            numpy numeric 2d array with the data to be tested.

        Returns
        -------
        TestResult
            Test result. Its fields depend on the specific test.

        """
        pass


@define
class HopkinsTestResult(TestResult):
    """Results of a Hopkins test.

    Attributes
    ----------
    rejectH0: bool
        True if the null hypothesis is rejected.
    pvalue: Number
        The p-value of the test.
    value: Number
        The value of the Hopkins statistic.

    """

    value: Number
    pvalue: Number


@define()
class HopkinsTest(StatTest):
    """Class to perform a Hopkins spatial randomness test.

    Attributes
    ----------
    n_iters : int, optional
        Number of iterations to perform the test. Final Hopkins statistic result
        is taken as the median of the results, by default is 100.
    sample_ratio : float, optional
        Sample ratio to take from the data, by default is ``0.1``. The number
        of samples is ``n*sample_ratio``.
    max_samples : int, optional
        Number of max samples to take from the data, by default is 100. If
        ``n_samples`` is greater than this value, it is set to this value.
    metric : Union[str, DistanceMetric], optional
        Metric to use for the distance between points, by default is 'euclidean'.
        Can be str or sklearn.neighbors.DistanceMetric.
    threshold : Number, optional
        Threshold to use with the Hopkins statistic value to define if H0 is rejected,
        by default is None. If set, it is used instead of the pvalue_threshold.
    pvalue_threshold : float, optional
        Threshold to use with the p-value to define if H0 is rejected, by default
        is 0.05.

    Notes
    -----
    The test compares the distance between a sample of ``m`` points ``X'``
    from the data set ``X`` and their nearest neighbors in ``X``, to the
    distances from ``X`` to their nearest neighbors in a uniform distribution.
    The null hypothesis is:

    *  H0: The dataset X comes from a Poisson Point Process.

    Which can be thought of as:

    *  H0: The dataset X does not present cluster structure.

    The formula to calculate the Hopkins statistic [1]_
    is ``h = sum(d1**l) / (sum(d1**l) + sum(d2**l))``, where:

    *  d1: distance_to_nearest_neighbor_in_X
    *  d2: distance_to_nearest_neighbor_in_uniform_distribution
    *  l: dimensionality of the data

    The Hopkins statistic is a number between 0.5 and 1. A value ``~0.5`` supports
    the null hypothesis. A value ``~1.0`` supports the alternative hypothesis.
    To get the p-value, the statistic is compared to a beta distribution with
    parameters ``(m, m)``.

    References
    ----------
    .. [1] Hopkins, B. and Skellam, J.G. (1954). A new method of determining the type of
         distribution of plant individuals”. Annals of Botany, 1954, 18(2),
         pp.213-227. https://doi.org/10.1093/oxfordjournals.aob.a083391

    Examples
    --------
    .. literalinclude:: ../../examples/stat_tests/hopkins.py
        :language: python
        :linenos:

    """

    sample_ratio: int = field(
        default=0.1, validator=[validators.gt(0), validators.le(1)]
    )
    max_samples: int = field(default=100)
    metric: Union[str, DistanceMetric] = field(default="euclidean")
    n_iters: int = field(default=100)
    # interpretation:
    # H0: data comes from uniform distribution
    # H1: data does not come from uniform distribution
    # if h = u/(u+w) ~ 1 => w = 0 luego hay estructura
    # if h = u/(u+w) ~ .5 => w ~ u luego no hay estructura
    # if h > .75 => reject H0, and in general  indicates a clustering
    # tendency at the 90% confidence level.
    threshold: Number = field(default=None)
    pvalue_threshold: float = field(default=0.05)

    def _get_pvalue(self, value: Number, n_samples: int):
        b = beta(n_samples, n_samples)
        if value > 0.5:
            # value is a random variate
            # that distributes as a beta(n, n)
            # that distribution is symmetric
            # around .5
            # let value = .5 +- e
            # get the probability
            # p of getting x ∈ (.5-4, .5+4)
            # As value close to .5 means support for
            # H0: data comes from uniform distribution
            # we want to get 1 - p
            # so we can compare with pvalue threshold
            return 1 - (b.cdf(value) - b.cdf(1 - value))
        else:
            return 1 - (b.cdf(1 - value) - b.cdf(value))

    @beartype
    def test(self, data: Numeric2DArray, *args, **kwargs):
        """Perform the Hopkins test.

        Parameters
        ----------
        data : Numeric2DArray
            Array containing the data.

        Returns
        -------
        HopkinsTestResult
            Test results.

        """
        obs, dims = data.shape

        n_samples = min(int(obs * self.sample_ratio), self.max_samples, obs)

        results = []
        for i in range(self.n_iters):
            sample = resample(data, n_samples=n_samples, replace=False)

            if self.metric == "mahalanobis":
                kwargs["V"] = np.cov(sample, rowvar=False)
            tree = BallTree(sample, leaf_size=2, metric=self.metric, *args, **kwargs)
            dist, _ = tree.query(sample, k=2)
            sample_nn_distance = dist[:, 1]

            max_data = data.max(axis=0)
            min_data = data.min(axis=0)
            uniform_sample = np.random.uniform(
                low=min_data, high=max_data, size=(n_samples, dims)
            )

            dist, _ = tree.query(uniform_sample, k=1)
            uniform_nn_distance = dist[:, 0]

            sample_sum = np.sum(sample_nn_distance**dims)
            uniform_sum = np.sum(uniform_nn_distance**dims)
            results.append(uniform_sum / (uniform_sum + sample_sum))

        value = np.median(np.array(results))
        pvalue = self._get_pvalue(value, n_samples)

        if self.threshold is not None:
            rejectH0 = value >= self.threshold
        else:
            rejectH0 = pvalue <= self.pvalue_threshold
        return HopkinsTestResult(value=value, rejectH0=rejectH0, pvalue=pvalue)


@define(auto_attribs=True)
class DipDistTestResult(TestResult):
    """Results of a dip dist test.

    Attributes
    ----------
    value : Number
        The value of the dip statistic.
    pvalue : Number
        The pvalue of the test.
    dist : Numeric1DArray
        The ordered distance array.

    """

    value: Number
    pvalue: Number
    dist: Numeric1DArray


@define(auto_attribs=True)
class DipDistTest(StatTest):
    """Class to perform a Dip-Dist test of multimodality over pairwise distances.

    The Dip-Dist implementation is based on the Python Dip test wrapper built by
    Ralph Ulrus, [2]_.

    Attributes
    ----------
    max_samples : int, optional
        Maximum number of samples to use, by default is ``1000``. If there are more
        data points than ``max_samples``, then the data is sampled.
    metric : Union[str, DistanceMetric], optional
        Metric to use for the distance between points, by default is 'euclidean'. Can be
        str or sklearn.neighbors.DistanceMetric.
    pvalue_threshold : float, optional
        Threshold to use with the p-value to define if H0 is rejected, by default
        is ``0.05``.

    Notes
    -----
    The test analyzes the pairwise distance distribution [3]_ between points
    in a data set to determine if said distribution is multimodal.
    The null hypothesis is:

    *  H0: The distance distribution is unimodal.

    Which can be thought of as:

    *  H0: The data set X does not present cluster structure.

    More specifically, the distance distribution will be unimodal
    for uniform data distributions or single cluster distributions.
    It will be multimodal when there are several clusters or when
    there is an aggregate of a uniform distribution and a cluster.
    The Hartigan's Dip statistic [4]_ can be defined as the maximum
    difference between an empirical distribution and its closest
    unimodal distribution calculated using the greatest convex minorant
    and the least concave majorant of the bounded distribution function.

    References
    ----------
    .. [3] R. Urlus (2022). A Python/C(++) implementation
         of Hartigan & Hartigan's dip test for unimodality.
         https://pypi.org/project/diptest/
    .. [2] A. Adolfsson, M. Ackerman, N. C. Brownstein (2018). To Cluster,
         or Not to Cluster: An Analysis of Clusterability Methods
         . https://doi.org/10.48550/arXiv.1808.08317
    .. [4] J. A. Hartigan and P. M. Hartigan (1985). The Dip Test of Unimodality.
         Annals of Statistics 13, 70–84. D
         OI: 10.1214/aos/1176346577

    Examples
    --------
    .. literalinclude:: ../../examples/stat_tests/dipdist.py
        :language: python
        :linenos:
    .. image:: ../../examples/stat_tests/dipdist.png

    """

    max_samples: int = 1000
    metric: str = "euclidean"
    pvalue_threshold: float = 0.05

    @beartype
    def test(self, data: Numeric2DArray, *args, **kwargs):
        """Perform the Dip-Dist test.

        Parameters
        ----------
        data : Numeric2DArray
            numpy 2d numeric array containing the data.

        Returns
        -------
        DipDistTestResult
            The test results.

        """
        obs, dims = data.shape

        if obs > self.max_samples:
            data = resample(data, n_samples=self.max_samples, replace=False)
        dist = np.ravel(np.tril(pairwise_distances(data, metric=self.metric)))
        dist = np.msort(dist[dist > 0])
        dip, pval = diptest(dist, sort_x=False, *args, **kwargs)
        rejectH0 = pval < self.pvalue_threshold
        return DipDistTestResult(value=dip, pvalue=pval, rejectH0=rejectH0, dist=dist)


@define(auto_attribs=True)
class RipleyKTestResult(TestResult):
    """Results of a Ripley's K test.

    Attributes
    ----------
    rejectH0: bool
        True if the null hypothesis is rejected.
    value: Number
        The value calculated to determine if H0 is rejected. If
        the test ``mode`` is ``chiu`` or ``ripley``, then the value is the
        the ``supremum`` statistic. If the test mode is ``ks``, then
        the value is the pvalue of the Kolmogorov-Smirnov test.
    radii: Numeric1DArray
        The radii used in the test.
    l_function: Numeric1DArray
        The L function values.

    """

    value: Number
    radii: Numeric1DArray
    l_function: Numeric1DArray


@define
class RipleysKTest(StatTest):
    """Class to perform the Ripleys K test of 2D spatial randomness.

    Attributes
    ----------
    rk_estimator : astropy.stats.RipleysKEstimator, optional
        Estimator to use for the Ripleys K function [5]_, by default
        is None. Only used if
        a custom RipleysKEstimator configuration is needed.

    mode : str, optional
        The comparison method to use to determine the rejection of H0, by default is
        "ripley". Allowed values are:

        #. "ripley": H0 rejected if ``s > ripley_factor * sqrt(area) / n`` where

            *   area: is the area of the 2D data set taken as a square window.
            *   n: is the number of points in the data set.
            *   ripley_factor: are the tabulated values calculated by Ripleys [5]_
                to determine p-value significance. Available Ripleys factors
                are ``p-value = 0.05`` -> ``factor = 1.42`` and
                ``p-value = 0.01`` -> ``factor = 1.68``.

        #. "chiu": H0 rejected if ``s > chiu_factor * sqrt(area) / n`` where:

            *   chiu_factor: are the tabulated values suggested by Chiu [6]_ to
                determine p-value significance. Available Chiu factors are
                ``p-value = 0.1 -> factor = 1.31``,
                ``p-value = 0.05 -> factor = 1.45`` and
                ``p-value = 0.01 -> factor = 1.75``.

        #. "ks": H0 rejected if
           ``kolmogorov_smirnov_test_pvalue < pvalue_threshold``, where
           kolmogorov_smirnov_test_pvalue is the p-value of the Kolmogorov Smirnov test
           comparing the estimated L function to the theoretical L function of a uniform
           distribution. This option is experimental and should be used with caution.

    radii : Numeric1DArray, optional
        numpy 1d numeric array containing the radii to use for the Ripleys K function,
        by default is None. If radii is None, radii are taken in a range
        ``[0, max_radius]``,
        where max_radius is calculated as:

            *  ``recommended_radius = short_side_of_rectangular_window / 4``
            *  ``recommended_radius_for_large_data_sets = sqrt(1000 / (pi * n))``
            *  ``max_radius = min(recommended_radius, recommended_radius_for_large_data_sets)``

        The steps between the radii values are calculated as
        ``step = max_radius / 128 / 4``.
        This procedure is the recommended one in R spatstat package [7]_.

    max_samples: int, optional
        The maximum number of samples to use for the test, by default is 5000. If the
        dataset has more than ``max_samples``, then the test is performed on a random sample
        of ``max_samples``.

    factor : float, optional
        The factor to use to determine the rejection of H0, by default is None. If factor is
        provided, then pvalue_threshold is ignored.

    Raises
    ------
    ValueError
        If tabulated factor for the chosen p-value threshold is not available, or if the
        chosen p-value threshold is invalid.

    Notes
    -----
    The test calculates the value of an estimate for the L function [8]_ (a
    form of Ripleys K function) for a set of radii taken from the
    center of the data set, and compares it to the theoretical L
    function of a uniform distribution. The null hypothesis is:

    *  H0: The data set X comes from a Poisson Point Process.

    Which can be thought of as:

    *  H0: The data set X does not present cluster structure.

    The Ripleys K(r) function is defined as the expected number
    of additional random points within a distance r of a typical random point.
    For a completely random point process (Poisson Point Process),
    ``K(r) = pi*r^2``. The L(r) function is a form of K(r) defined as
    ``L(r) = sqrt(K(r)/pi)``.
    The statistic to define if H0 is rejected based on the L function is the
    ``supremum`` of the difference ``s = max(L(r) - r)``.

    References
    ----------
    .. [5] B. D. Ripley (1979). Tests of Randomness for Spatial Point Patterns.
         J. R. Statist. Soc. B (1979), 41, No.3, pp. 368-374.
         https://doi.org/10.1111/j.2517-6161.1979.tb01091.x
    .. [6] S. N. Chiu (2007). Correction to Koen's critical values in testing
         spatial randomness. Journal of Statistical Computation and Simulation
         2007 77(11-12):1001-1004. DOI: 10.1080/10629360600989147
    .. [7] A. Baddeley, R. Turner (2005). Spatstat: An R Package for Analyzing
         Spatial Point Patterns. Journal of Statistical Software, 12(6), 1–42.
         DOI: 10.18637/jss.v012.i06.
    .. [8] J. Besag (1977). Contribution to the Discussion on Dr. Ripley’s
         Paper. Journals of the Royal Statistical Society, B39, 193-195.

    Examples
    --------
    .. literalinclude:: ../../examples/stat_tests/ripley.py
        :language: python
        :linenos:
    .. image:: ../../examples/stat_tests/ripley.png

    """  # noqa: E501

    rk_estimator: RipleysKEstimator = None

    _used_fitted_rk_estimator: RipleysKEstimator = None

    _scaler: TransformerMixin = MinMaxScaler()

    _ripley_factors = {
        0.05: 1.42,
        0.01: 1.68,
    }

    _chiu_factors = {
        0.1: 1.31,
        0.05: 1.45,
        0.01: 1.75,
    }

    mode: str = field(
        validator=validators.in_(["ripley", "chiu", "ks"]), default="ripley"
    )

    radii: Numeric1DArray = None

    factor: float = None

    pvalue_threshold: float = field(default=0.05)

    max_samples: int = field(default=5000)

    @pvalue_threshold.validator
    def _check_pvalue_threshold(self, attribute, value):
        if self.factor is None:
            if self.mode == "ripley" and value not in self._ripley_factors.keys():
                raise ValueError(
                    f"{value} is not a valid pvalue threshold for {self.mode} rule."
                    f" Must be one of {self._ripley_factors.keys()}"
                )
            elif self.mode == "chiu" and value not in self._chiu_factors.keys():
                raise ValueError(
                    f"{value} is not a valid pvalue threshold for {self.mode} rule."
                    f" Must be one of {self._chiu_factors.keys()}"
                )
        elif value <= 0 or value >= 1:
            raise ValueError(
                f"{value} is not a valid pvalue threshold. Must be between 0 and 1."
            )

    def _empirical_csr_rule(
        self, l_function: Numeric2DArray, radii: Numeric2DArray, area: Number, n: int
    ):
        supremum = np.max(np.abs(l_function - radii))
        if self.factor:
            factor = self.factor
        elif self.mode == "ripley":
            factor = self._ripley_factors[self.pvalue_threshold]
        else:
            factor = self._chiu_factors[self.pvalue_threshold]
        return supremum, supremum >= factor * np.sqrt(area) / float(n)

    def _ks_rule(self, l_function: Numeric2DArray, radii: Numeric2DArray):
        pvalue = ks_2samp(l_function, radii).pvalue
        return pvalue, pvalue <= self.pvalue_threshold

    @beartype
    def test(self, data: Numeric2DArray, *args, **kwargs):
        """Perform the Ripleys K test of 2D spatial randomness.

        Parameters
        ----------
        data : Numeric2DArray
            numpy 2d numeric array containing the data set to test.

        Returns
        -------
        RipleysKTestResult
            Result of the Ripleys K test.


        Warns
        -----
        UserWarning
            Warns if some dataset points are repeated (exactly equal). In that case,
            the RipleysKEstimator will not be able to calculate the L function,
            so repeated points will will be eliminated before the test. Bound to
            change when RipleysKEstimator implementation is changed.

        """
        data_unique = np.unique(data, axis=0)
        if data_unique.shape[0] != data.shape[0]:
            warn(
                "There are repeated data points that cause"
                " astropy.stats.RipleysKEstimator to fail, they will be removed.",
                category=UserWarning,
            )
            data = data_unique

        if data.shape[0] > self.max_samples:
            data = resample(data, n_samples=self.max_samples, replace=False)

        obs, dims = data.shape

        # negative data values are not handled properly by RipleysKEstimator
        # hence, it seems that MinMax scaling is a must
        # in the future, an alternative RipleysKEstimator implementation could be used
        if self._scaler is not None:
            data = self._scaler.fit_transform(data)

        x_min = data[:, 0].min()
        x_max = data[:, 0].max()
        y_min = data[:, 1].min()
        y_max = data[:, 1].max()

        if self.radii is None:
            # considers rectangular window
            # based on R spatstat rmax.rule
            short_side = min(x_max - x_min, y_max - y_min)
            radii_max_ripley = short_side / 4
            radii_max_large = np.sqrt(1000 / (np.pi * obs))
            radii_max = min(radii_max_ripley, radii_max_large)
            step = radii_max / 128 / 4
            radii = np.arange(0, radii_max + step, step)
        else:
            radii = self.radii

        if self.rk_estimator is None:
            # Could be extended to other shapes
            # depending on the edge correction methods
            # available. Could use ConvexHull to get the
            # area.
            area = (x_max - x_min) * (y_max - y_min)
            self._fitted_rk_estimator = RipleysKEstimator(
                area=area,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
        else:
            self._fitted_rk_estimator = deepcopy(self.rk_estimator)
            area = self._fitted_rk_estimator.area

        if kwargs.get("mode") is None:
            # Best mode for rectangular window
            kwargs["mode"] = "ripley"

        l_function = self._fitted_rk_estimator.Lfunction(data, radii, *args, **kwargs)

        if self.mode == "ks":
            value, rejectH0 = self._ks_rule(l_function, radii)
        else:
            value, rejectH0 = self._empirical_csr_rule(l_function, radii, area, obs)

        return RipleyKTestResult(
            value=value, rejectH0=rejectH0, radii=radii, l_function=l_function
        )
