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

"""Module for useful statistical tests."""

from abc import abstractmethod
from numbers import Number
from typing import Union
from warnings import warn

import numpy as np
from astropy.stats import RipleysKEstimator
from attrs import define, field, validators
from beartype import beartype
from beartype.vale import IsAttr, IsEqual
from diptest import diptest
from numpy.typing import NDArray
from scipy.stats import ks_2samp, beta
from sklearn.base import TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree, DistanceMetric
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample


from typing_extensions import Annotated


Numeric2DArray = Annotated[NDArray[np.number], IsAttr["ndim", IsEqual[2]]]
Numeric1DArray = Annotated[NDArray[np.number], IsAttr["ndim", IsEqual[1]]]


@define(auto_attribs=True)
class TestResult:
    """Base class to hold the results of a statistical test."""

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
            Test result. Its fields depend on the specific test, but always
            has field rejectH0: boolean
        """
        pass


@define(auto_attribs=True)
class HopkinsTestResult(TestResult):
    """Results of a Hopkins test."""

    value: Number
    pvalue: Number


@define(auto_attribs=True)
class RipleyKTestResult(TestResult):
    """Results of a Ripley's K test."""

    value: Number
    radii: Numeric1DArray
    l_function: Numeric1DArray


@define(auto_attribs=True)
class DipDistTestResult(TestResult):
    """Results of a dip dist test."""

    value: Number
    pvalue: Number
    dist: Numeric1DArray


@define(auto_attribs=True)
class HopkinsTest(StatTest):
    """Class to perform a Hopkins spatial randomness test.

    Compares the distance between a sample of m points X' from the data set X and
    their nearest neighbors in X, to the distances from X to their nearest neighbors
    in a uniform distribution. The null hypothesis is:
        H0: The dataset X comes from a Poisson Point Process.
    Which can be thought of as:
        H0: The dataset X does not present cluster structure.
    The formula to calculate the Hopkins statistic is:
        h = sum(d1**l) / (sum(d1**l) + sum(d2**l))
    where:
        d1: distance_to_nearest_neighbor_in_X
        d2: distance_to_nearest_neighbor_in_uniform_distribution
        l: dimensionality of the data
    The Hopkins statistic is a number between 0.5 and 1. A value ~ 0.5 supports
    the null hypothesis. A value ~ 1.0 supports the alternative hypothesis.
    To get the p-value, the statistic is compared to a beta distribution with
    parameters (m, m).

    Attributes
    ----------
    n_iters : int
        Number of iterations to perform the test. Final Hopkins statistic result
        is taken as the median of the results, by default is 100.
    n_samples : int
        Number of samples to take from the data, by default is 0.1*n where n is
        the number of points in the data set, as it is the recommended value.
    metric : Union[str, DistanceMetric]
        Metric to use for the distance between points, by default is 'euclidean'.
        Can be str or sklearn.neighbours.DistanceMetric.
    threshold : Number, optional
        Threshold to use with the Hopkins statistic value to define if H0 is rejected,
        by default is None. If set, it is used instead of the pvalue_threshold.
    pvalue_threshold : float
        Threshold to use with the p-value to define if H0 is rejected, by default
        is 0.05.

    """

    n_samples: int = None
    metric: Union[str, DistanceMetric] = "euclidean"
    n_iters: int = 100
    # interpretation:
    # H0: data comes from uniform distribution
    # H1: data does not come from uniform distribution
    # if h = u/(u+w) ~ 1 => w = 0 luego hay estructura
    # if h = u/(u+w) ~ .5 => w ~ u luego no hay estructura
    # if h > .75 => reject H0, and in general  indicates a clustering
    # tendency at the 90% confidence level.
    threshold: Number = None
    pvalue_threshold: float = 0.05

    def _get_pvalue(self, value: Union[Number, DipDistTestResult], n_samples: int):
        b = beta(n_samples, n_samples)
        if value > 0.5:
            return 1 - (b.cdf(value) - b.cdf(1 - value))
        else:
            return 1 - (b.cdf(1 - value) - b.cdf(value))

    @beartype
    def test(self, data: Numeric2DArray, *args, **kwargs):
        """Perform the Hopkins test.

        Parameters
        ----------
        data : Numeric2DArray
            numpy 2d numeric array containing the data.

        Returns
        -------
        HopkinsTestResult
            Result containing:
                - value: the Hopkins statistic
                - pvalue: the p-value
                - rejectH0: True if H0 is rejected, False otherwise.

        """
        obs, dims = data.shape

        if self.n_samples is None:
            n_samples = int(obs * 0.1)
        else:
            n_samples = min(obs, self.n_samples)

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
class DipDistTest(StatTest):
    """Class to perform a Dip-Dist test of multimodality over pairwise distances.

    It analyzes the distribution of distances between pairs of points in a data set
    to determine if the data set is multimodal. The null hypothesis is:
        H0: The distance distribution is unimodal.
    Which can be thought of as:
        H0: The data set X does not present cluster structure.
    Hartigan's Dip statistic is the maximum difference between an empirical distribution
    and its closest unimodal distribution calculated using the greatest convex minorant
    and the least concave majorant of the bounded distribution function.

    Attributes
    ----------
    n_samples : int
        number of samples to take from the data, by default is min(n, 100)
        where n is the number of points in the data set. This value is simply chosen as
        to reduce the computation time.
    metric : Union[str, DistanceMetric]
        Metric to use for the distance between points, by default is 'euclidean'. Can be
        str or sklearn.neighbours.DistanceMetric.
    pvalue_threshold : float
        Threshold to use with the p-value to define if H0 is rejected, by default
        is 0.05.

    """

    n_samples: int = None
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
            The result containing:
                - value: the Dip statistic
                - pvalue: the p-value
                - rejectH0: True if H0 is rejected, False otherwise.

        """
        obs, dims = data.shape

        if self.n_samples is not None:
            n_samples = min(obs, self.n_samples)
        else:
            n_samples = obs

        sample = resample(data, n_samples=n_samples, replace=False)
        dist = np.ravel(np.tril(pairwise_distances(sample, metric=self.metric)))
        dist = np.msort(dist[dist > 0])
        dip, pval = diptest(dist, *args, **kwargs)
        rejectH0 = pval < self.pvalue_threshold
        return DipDistTestResult(value=dip, pvalue=pval, rejectH0=rejectH0, dist=dist)


@define
class RipleysKTest(StatTest):
    """Class to perform the Ripleys K test of 2D spatial randomness.

    It calculates the value of an estimate for the L function (a form of Ripleys
    K function) for a set of radii taken from the center of the data set,
    and compares it to the theoretical L function of a uniform distribution where
    L_function(radii) = radii. The null hypothesis is:
        H0: The data set X comes from a Poisson Point Process.
    Which can be thought of as:
        H0: The data set X does not present cluster structure.
    The Ripleys K function is defined as:
    The L function is defined as:
        L(r) = sqrt(K(r)/pi)
    The statistic to define if H0 is rejected is:
        s = max(L(r) - r)

    Attributes
    ----------
    rk_estimator : astropy.stats.RipleysKEstimator, optional
        Estimator to use for the Ripleys K function, by default is None. Only used if
        a custom RipleysKEstimator configuration is needed.

    mode: str, optional
        The comparison method to use to determine the rejection of H0, by default is
        'ripleys'.
        Allowed values are:
            - 'ripleys': H0 rejected if s > ripley_factor * sqrt(area) / n
                where:
                    - area: is the area of the 2D data set taken as a square window.
                    - n: is the number of points in the data set.
                    - ripley_factor: are the tabulated values calculated by Ripleys to
                    determine p-value significance. Available Ripleys factors are:
                            - p-value = 0.05 -> factor = 1.42
                            - p-value = 0.01 -> factor = 1.68
            - 'chiu': H0 rejected if s > chiu_factor * sqrt(area) / n
                where:
                    - chiu_factor: are the tabulated values calculated by Chiu to
                    determine p-value significance. Available Chiu factors are:
                            - p-value = 0.1 -> factor = 1.31
                            - p-value = 0.05 -> factor = 1.45
                            - p-value = 0.01 -> factor = 1.75
            - 'ks': H0 rejected if kolmogorov_smirnov_test_pvalue < pvalue_threshold
                where:
                    - kolmogorov_smirnov_test_pvalue: is the p-value of the Kolmogorov
                    Smirnov test comparing the L function to the theoretical L function.
                This option is experimental and should be used with caution.

    radii : Numeric1DArray, optional
        numpy 1d numeric array containing the radii to use for the Ripleys K function,
        by default is None. If radii is None, radii are taken in a range [0, max_radius]
        where max_radius is calculated as:
            recommended_radius = short_side_of_rectangular_window / 4
            recommended_radius_for_large_data_sets = sqrt(100 / pi * n)
            max_radius = min(recommended_radius, recommended_radius_for_large_data_sets)

        The steps between the radii values are calculated as:
            step = max_radius / 128 / 4
        This procedure is the recommended one in R spatstat package.

    Raises
    ------
    ValueError
        If tabulated factor for the chosen p-value threshold is not available, or if the
        chosen p-value threshold is invalid.

    """

    rk_estimator: RipleysKEstimator = None

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
            Result of the Ripleys K test, containing the following attributes:
                pvalue: float
                    The p-value of the test.
                rejectH0: bool
                    True if H0 is rejected, False otherwise.
                radii: Numeric1DArray
                    numpy 1d numeric array containing the radii used for the test.
                l_function: Numeric1DArray
                    numpy 1d numeric array containing the L function values.

        Warnings
        --------
        UserWarning
            Warns if some dataset points are repeated exactly. In that case,
            the RipleysKEstimator will not be able to calculate the L function,
            so repeated points will will be eliminated before the test. Bound to
            change when RipleysKEstimator implementation is changed.

        """
        data_unique = np.unique(data, axis=0)
        if data_unique.shape[0] != data.shape[0]:
            warn(
                "There are repeated data points that cause"
                " astropy.stats.RipleysKEstimator to break, they will be removed.",
                category=UserWarning,
            )
            data = data_unique

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

        if self.rk_estimator is None:
            # Could be extended to other shapes
            # depending on the edge correction methods
            # available. Could use ConvexHull to get the
            # area.
            area = (x_max - x_min) * (y_max - y_min)
            self.rk_estimator = RipleysKEstimator(
                area=area,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
        else:
            area = self.rk_estimator.area

        if kwargs.get("mode") is None:
            # Best mode for rectangular window
            kwargs["mode"] = "ripley"

        l_function = self.rk_estimator.Lfunction(data, radii, *args, **kwargs)

        if self.mode == "ks":
            value, rejectH0 = self._ks_rule(l_function, radii)
        else:
            value, rejectH0 = self._empirical_csr_rule(l_function, radii, area, obs)

        return RipleyKTestResult(
            value=value, rejectH0=rejectH0, radii=radii, l_function=l_function
        )
