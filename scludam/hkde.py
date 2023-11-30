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

"""Module for Kernel Density Estimation with variable bandwidth matrices.

The module provides a class for multivariate Kernel Density Estimation with a bandwidth
matrix per observation. Such matrices are created from a baseline bandwidth calculated
from the Plugin or Rule Of Thumb (scott or silverman) methods. Variable errors and
covariances can be added to the matrices.

"""

from abc import abstractmethod
from numbers import Number
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from attrs import define, field, validators
from beartype import beartype
from itertools import product
from rpy2.robjects import r
from scipy.stats import multivariate_normal
from statsmodels.stats.correlation_tools import corr_nearest, cov_nearest

from scludam.plots import bivariate_density_plot, univariate_density_plot
from scludam.rutils import (
    assign_r_args,
    clean_r_session,
    disable_r_console_output,
    disable_r_warnings,
    load_r_packages,
)
from scludam.type_utils import (
    ArrayLike,
    Numeric1DArray,
    Numeric1DArrayLike,
    Numeric2DArray,
    NumericArray,
    OptionalNumeric1DArrayLike,
    OptionalNumeric2DArray,
    OptionalNumericArray,
    _type,
)

disable_r_warnings()
disable_r_console_output()
load_r_packages(r, ["ks"])


class BandwidthSelector:
    """Base class for bandwidth selector."""

    @abstractmethod
    def get_bw(data: Numeric2DArray, *args, **kwargs):
        """Get the bandwidth for the given data.

        Parameters
        ----------
        data : Numeric2DArray
            Data.

        """
        pass


@define
class PluginSelector(BandwidthSelector):
    """Bandwidth selector based on the Plugin method.

    It uses the Plugin method with unconstraned pilot bandwidth
    [1]_ [2]_ implementation in the
    `ks` R package [3]_. See the `ks` package documentation for
    more information on
    parameter values. All attributes are passed
    to ``ks::Hpi`` function.

    Attributes
    ----------
    nstage : int, optional
        Number of calculation stages, can be
        1 or 2, by default 2.
    pilot: str, optional
        Kind of pilot bandwidth.
    binned: bool, optional
        Use binned estimation, by default False.
    diag: bool, optional
        Whether to use the diagonal bandwidth,
        by default False. If true, ``ks::Hpi.diag``
        is used.

    References
    ----------
    .. [1] Chacon, J.E., Duong, T. (2010) Multivariate plug-in bandwidth
        selection with unconstrained pilot
        matrices. Test, 19, 375-398.
    .. [2] Chacon, J.E., Duong, T. (2018) Multivariate Kernel Smoothing
        and Its Applications (pp. 43-66).
    .. [3] Duong, T. (2013). ks: Kernel Smoothing. R package version 1.13.3.
        https://CRAN.R-project.org/package=ks

    """

    nstage: int = None
    pilot: str = None
    binned: bool = None
    diag: bool = False

    def _build_r_command(self, data: Numeric2DArray):
        params = {
            "nstage": self.nstage,
            "pilot": self.pilot,
            "binned": self.binned,
        }
        # delete all previous session variables
        clean_r_session(r, "var")
        _, rparams = assign_r_args(r, x=data, **params)
        return f'ks::Hpi{".diag" if self.diag else ""}({rparams})'

    @beartype
    def get_bw(self, data: Numeric2DArray):
        """Get the bandwidth for the given data.

        Builds R ``ks::Hpi`` command and executes it.

        Parameters
        ----------
        data : Numeric2DArray
            Data.

        Returns
        -------
        Numeric2DArray
            Optimal bandwidth matrix H acoording to the
            Plugin Method.

        """
        _, dims = data.shape
        command = self._build_r_command(data)
        result = r(command)
        return np.asarray(result)


@define
class RuleOfThumbSelector(BandwidthSelector):
    """Bandwidth selector based on the Rule of Thumb method.

    Attributes
    ----------
    rule : str, optional
        Name of the rule of thumb to use, by default "scott".
        Can be "scott" or "silverman".

    diag : bool, optional
        Whether to use the diagonal bandwidth,
        by default False.

    Raises
    ------
    ValueError
        If rule is not "scott" or "silverman".

    """

    rule: str = field(default="scott", validator=validators.in_(["scott", "silverman"]))
    diag: bool = False

    def _scotts_factor(self, data):
        n = data.shape[0]
        d = data.shape[1]
        return np.power(n, -1.0 / (d + 4))

    def _silverman_factor(self, data):
        n = data.shape[0]
        d = data.shape[1]
        return np.power(n * (d + 2.0) / 4.0, -1.0 / (d + 4))

    def _get_factor(self, data):
        if self.rule == "scott":
            return self._scotts_factor(data)
        elif self.rule == "silverman":
            return self._silverman_factor(data)
        else:
            raise ValueError("Invalid rule")

    def _get_data_covariance(self, data, weights: Union[Numeric2DArray, None]):
        kws = {}
        if weights is not None:
            kws["aweights"] = weights
        if self.diag:
            data_covariance = np.diagflat(
                [
                    np.cov(data[:, i], rowvar=False, bias=False, **kws)
                    for i in range(data.shape[1])
                ]
            )
        else:
            data_covariance = np.cov(data, rowvar=False, bias=False, **kws)
        data_covariance = cov_nearest(data_covariance)
        return data_covariance

    @beartype
    def get_bw(self, data: Numeric2DArray, weights: Union[None, Numeric1DArray] = None):
        """Calculate bandwith matrix using the rule of thumb.

        Parameters
        ----------
        data : Numeric2DArray
            Data to be used
        weights : Union[None, Numeric1DArray]
            Optional weights to be used.

        """
        # This function does not consider the case
        # of weights being zero, so they should be at least > 1e-08.
        data_covariance = self._get_data_covariance(data, weights)
        factor = self._get_factor(data)
        return data_covariance * factor**2


@define
class HKDE:
    """Kernel Density Estimation with variable bandwidth matrices (H).

    Only for multivariate data (2d-nd). As it does not uses
    KDE by Fast Fourier Transform, it may take some time for
    big datasets.

    Attributes
    ----------
    bw: Union[BandwidthSelector, Number, NumericArray], optional
        Bandwidth to be used, by default an instance
        of :class:`~scludam.hkde.PluginSelector`. It can be:

        *  an instance of :class:`~scludam.hkde.BandwidthSelector`:
           the base bandwidth is calculated using the given selector.
        *  a Number: the base bandwidth is calculated as a diagonal
           matrix with the given value.
        *  an Array: the base bandwidth is taken as the given
           array. The array shape must be (n, d, d) where n is the
           number of observations and d is the number of dimensions.
        *  a String: the name of the rule of thumb to be used. One of
              "scott" or "silverman".
    error_convolution: bool, optional
        When true:
        *  It can only estimate density for the same points as the data.
        That is, eval points are equal to data points.
        *  It always is a leave-one-out estimation.
        *  To calculate the contribution of point A to the density
        evaluated at point B, both the bandwidth matrix of point A and
        the bandwidth matrix of point B are convolved.
        *  This option should be used to get an accurate measure of the
        density at the data points considering the uncertainty of all
        points, themselves included.
        *  As a new matrix is calculated for each combination of points,
        is the slowest option. Although it has been optimized with ball
        tree to reduce the number of matrices used, it could be
        problematic for big concentrated datasets.
        *  Default is False.
    
    Examples
    --------
    .. literalinclude:: ../../examples/hkde/hkde.py
        :language: python
        :linenos:
    .. image:: ../../examples/hkde/hkde.png

    """

    # input attrs
    bw: Union[BandwidthSelector, Number, NumericArray, List[Number], str] = field(
        default=PluginSelector(),
        validator=_type(
            Union[BandwidthSelector, Number, List[Number], NumericArray, str]
        ),
    )
    error_convolution: bool = False

    # internal attrs
    _kernels: ArrayLike = None
    _weights: Numeric1DArrayLike = None
    _covariances: NumericArray = None
    _base_bw: NumericArray = None
    _data: Numeric2DArray = None
    _n: int = None
    _d: int = None
    _n_eff: int = None
    _eff_mask: ArrayLike = None
    _maxs: Numeric1DArray = None
    _mins: Numeric1DArray = None

    @beartype
    def set_weights(self, weights: ArrayLike):
        """Set the weights for each data point.

        Set a weight value for each data point,
        between 0 and 1.

        Parameters
        ----------
        weights : ArrayLike
            Weights for each data point.

        Returns
        -------
        HKDE
            Instance of :class:`~scludam.hkde.HKDE`.

        Raises
        ------
        ValueError
            If weights are not between 0 and 1 or do
            not match the correct dimensions (Array
            of shape (n,)).

        """
        weights = np.asarray(weights)
        if len(weights.shape) != 1:
            raise ValueError("Weights must be 1d np ndarray.")
        if np.any(weights > 1):
            raise ValueError("Weight values must belong to [0,1].")
        if weights.shape[0] != self._n:
            raise ValueError("Data must have same n as weights.")
        self._weights = weights
        # default atol used by numpy isclose is 1e-08
        self._eff_mask = self._weights > 1e-08
        self._n_eff = np.sum(self._weights[self._eff_mask])
        return self

    def _get_err_matrices(self, err: Numeric2DArray, corr: NumericArray = None):
        if err.shape != (self._n, self._d):
            raise ValueError("error array must have the same shape as data array.")
        if corr is None:
            return np.apply_along_axis(lambda x: np.diag(x**2), -1, err)
        corr_matrices = self._get_corr_matrices(corr)
        return (
            np.apply_along_axis(lambda x: x * np.atleast_2d(x).T, -1, err)
            * corr_matrices
        )

    def _get_corr_matrices(self, corr: NumericArray):
        n, d = (self._n, self._d)
        # correlation is given
        # is array
        if corr.shape == (d, d):
            # correlation is given as global correlation matrix per dims
            # first try to find nearest correct correlation
            # it ensures matrix is positive semidefinite
            corr = corr_nearest(corr)
            if not np.allclose(np.diag(corr), np.ones(d)):
                raise ValueError("Correlation matrix must have 1 in diagonal")
            return np.repeat(corr, repeats=n, axis=1).reshape((d, d, n)).T
        elif corr.shape == (n, int(d * (d - 1) / 2)):
            # correlation is given per observation per obs, per dims
            # pairwise corr coef
            # (no need for the 1s given by corr(samevar, samevar))
            # per observation. Example: for 1 obs and 4 vars, lower triangle of corr
            # matrix looks like:
            # 12
            # 13 23
            # 14 24 34
            # method should receive obs1 => [12, 13, 23, 14, 24, 34]
            n_corrs = corr.shape[1]
            corrs = np.zeros((n, d, d))
            tril_idcs = tuple(
                map(
                    tuple,
                    np.vstack(
                        (
                            np.arange(n).repeat(n_corrs),
                            np.tile(
                                np.array(np.tril_indices(d, k=-1)),
                                (n,),
                            ),
                        )
                    ),
                )
            )
            corrs[tril_idcs] = corr.ravel()
            corrs = corrs + np.transpose(corrs, (0, 2, 1))
            diag_idcs = tuple(
                map(
                    tuple,
                    np.vstack(
                        (
                            np.arange(n).repeat(d),
                            np.tile(np.array(np.diag_indices(d)), (n,)),
                        )
                    ),
                )
            )
            corrs[diag_idcs] = 1
            return corrs
        else:
            raise ValueError("Wrong corr dimensions")

    def _get_bw_matrices(self, data: Numeric2DArray):
        if isinstance(self.bw, BandwidthSelector):
            bw_matrix = self.bw.get_bw(data[self._eff_mask])
        elif isinstance(self.bw, str):
            bw_matrix = RuleOfThumbSelector(rule=self.bw).get_bw(
                data[self._eff_mask], self._weights[self._eff_mask]
            )
        elif isinstance(self.bw, np.ndarray) or isinstance(self.bw, list):
            if isinstance(self.bw, list):
                self.bw = np.array(self.bw)
            if len(self.bw.shape) == 1 and self.bw.shape[0] == self._d:
                bw_matrix = np.diag(self.bw)
            elif self.bw.shape == (self._d, self._d):
                bw_matrix = self.bw
            else:
                raise ValueError("Incorrect shape of bandwidth array")
        self._base_bw = bw_matrix
        return np.repeat(bw_matrix[:, np.newaxis], self._n, 1).swapaxes(0, 1)

    def _get_cov_matrices(
        self,
        data: Numeric2DArray,
        err: Numeric2DArray = None,
        corr: NumericArray = None,
    ):
        bw_matrices = self._get_bw_matrices(data)
        if err is None and corr is None:
            return bw_matrices
        err_matrices = self._get_err_matrices(err, corr)
        # sum of covariance matrices convolves a kernel for bw and a kernel for error
        return bw_matrices + err_matrices

    @beartype
    def fit(
        self,
        data: Numeric2DArray,
        err: OptionalNumeric2DArray = None,
        corr: OptionalNumericArray = None,
        weights: OptionalNumeric1DArrayLike = None,
        *args,
        **kwargs,
    ):
        """Fit a KDE model to the provided data.

        Creates covariances matrices and kernel
        instances.

        Parameters
        ----------
        data : Numeric2DArray
            Data.
        err : OptionalNumeric2DArray, optional
            Error array of shape (n, d), by default None.
            Errors are added to the base bandwidth matrix
            to create individual H matrices per datapoint.
        corr : OptionalNumericArray, optional
            Correlation coeficients, by default None.
            Coeficients are added to the base bandwith matrix
            to create individual H matrices per datapoint.
            Can be one of:

            *  NumericArray of shape (d, d): global correlation
               matrix. Applied in every bandwidth matrix H.
            *  Numeric2DArray of shape (n, (d * (d - 1) / 2):
               individual correlation matrices. Each column of
               the array represents a correlation between two
               variables, for all observations. Order of columns
               must follow a lower triangle matrix. For example:
               for four variables, lower triangle of corr
               matrix looks like:

               .. code-block:: python

                    corr(v1, v2)
                    corr(v1, v3), corr(v2, v3)
                    corr(v1, v4), corr(v2, v4), corr(v3, v4)

               So a valid ``corr`` array for two datapoints would be:

               .. code-block:: python

                    np.array([
                        [ corr1(v1, v2), corr1(v1, v3), corr1(v2, v3),
                          corr1(v1, v4), corr1(v2, v4), corr1(v3, v4) ],
                        [ corr2(v1, v2), corr2(v1, v3), corr2(v2, v3),
                          corr2(v1, v4), corr2(v2, v4), corr2(v3, v4) ],
                    ])

        weights : OptionalNumeric1DArrayLike, optional
            Weights to be used for each data point, by default None.
            If None, all datapoints have the same
            weight.

        Returns
        -------
        HKDE
            Instance of :class:`~scludam.hkde.HKDE`.

        Notes
        -----
        Base bandwidth matrix is calculated from the ``bw`` parameter. If no additional
        parameters are provided, the base bandwidth matrix is
        used for all datapoints. If ``err`` and/or ``corr`` are provided, they are
        used to create individual covariance matrices for each datapoint [5]_. The final
        matrix used for each kernel is the sum of the base matrix and the individual
        covariance matrix, which is equivalent to convolving two gaussian kernels, one
        for the base bandwidth matrix and one for the individual covariance matrix.
        The base bandwidth is considered as the minimum bandwidth of the KDE process,
        for a data point without uncertainty, while the final matrix incorporates the
        uncertainty if provided.

        References
        ----------
        .. [5] Luri, X. et al. (2018). Gaia Data Release 2: using Gaia parallaxes.
            Astronomy and Astrophysics, 616, A9. doi: 10.1051/0004-6361/201832964

        """  # noqa E501
        self._n, self._d = data.shape
        self._data = data
        self._maxs = data.max(axis=0)
        self._mins = data.min(axis=0)

        weights = weights if weights is not None else np.ones(self._n)
        self.set_weights(weights)

        self._covariances = self._get_cov_matrices(data, err, corr)

        self._kernels = np.array(
            [
                multivariate_normal(
                    data[i],
                    self._covariances[i],
                    *args,
                    **kwargs,
                )
                for i in range(self._n)
            ]
        )
        return self

    def _is_fitted(self):
        return not (
            self._kernels is None
            or self._weights is None
            or self._n_eff is None
            or self._eff_mask is None
            or self._covariances is None
        )

    def _calculate_biggest_hypersphere(self):
        # If sum of diagonal is bigger when correlations are small, then matrix is bigger
        # get the self._covariances matrix which diagonal sums the biggest
        sums = np.array([np.diagonal(cc).sum() for cc in self._covariances])
        # get the 99 percentile of the sums
        biggest_cov = np.percentile(sums, 99)
        closest_cov = np.argmin(np.abs(sums - biggest_cov))
        # get the index of the biggest matrix
        biggest_matrix = self._covariances[closest_cov]
        # get the biggest matrix
        # create a multivariate normal around 0 with the biggest matrix, accounting for dims
        biggest_kde = multivariate_normal(
            np.zeros(self._d),
            biggest_matrix,
        )
        # determine where the pdf is <= 1e-08 in all dimensions
        # and take the distance between 0 and that point
        # as the radius of the biggest sphere
        grid_range = (-3, 3)
        resolution = 10
        threshold = 1e-08
        grid_linspace = np.linspace(grid_range[0], grid_range[1], resolution)
        dim = self._d
        points = np.array(list(product(grid_linspace, repeat=dim)))
        pdf_values = biggest_kde.pdf(points)
        points_above_threshold = points[pdf_values > threshold]
        distances = np.linalg.norm(points_above_threshold, axis=1)
        max_distance = np.min(distances)
        return max_distance

    def _build_tree_ball(self, radius: float, neighbours: Numeric2DArray, eval_points: Numeric2DArray):
        from sklearn.neighbors import BallTree
        # build a ball tree with the data
        tree = BallTree(neighbours)
        # get the indexes of the points that are inside the ball
        return tree.query_radius(eval_points, radius)

    def _pdf_with_error_convolution(self):
        if not self._is_fitted():
            raise Exception("Model not fitted. Try excecuting fit function first.")
        eval_points = np.asarray(self._data)
        obs, dims = eval_points.shape
        if dims != self._d:
            raise ValueError("Eval points must have same dims as data.")
        if obs < 1:
            raise ValueError("Eval points cannot be empty")

        if self._n_eff <= 0:
            return np.zeros(obs)

        pdf = np.zeros(obs)
        all_covariances = self._covariances
        weights = self._weights[self._eff_mask]
        covariances = self._covariances[self._eff_mask]
        data = self._data[self._eff_mask]
        n = self._n_eff

        tree = self._build_tree_ball(self._calculate_biggest_hypersphere(), data, self._data)

        # put weights and normalization toghether in each step
        # pdf(point) = sum(ki(point)*wi/(sum(w)-wi))
        norm_weigths = weights / (n - weights)
        pdf = np.zeros(self._n)
        for j, p in enumerate(eval_points):
            # print(f'hkde progress: {round(j/self._n * 100, 2)}')
            # get the indexes of the points that are inside the ball
            indexes = tree[j]
            # get the covariances of the points that are inside the ball
            point_cov = all_covariances[j]
            applied_ks = 0
            for idx in indexes:
                mean = data[idx]
                if not np.allclose(mean, p):
                    cov = covariances[idx] + point_cov
                    k = multivariate_normal(
                        mean,
                        cov
                    )
                    tosum = k.pdf(p) * norm_weigths[idx]
                    applied_ks += tosum
            pdf[j] = applied_ks

        if obs == 1:
            # return as float value
            return pdf[0]
        return pdf

    def pdf(self, eval_points: Numeric2DArray, leave1out: bool = True):
        """Probability density function.

        Evaluate the probability density function at the provided
        points, using the fitted KDE model.

        Parameters
        ----------
        eval_points : Numeric2DArray
            Observation or observations to evaluate the PDF at.
        leave1out : bool, optional
            Wether to set weigth to 0 for the point being evaluated,
            by default True.

        Returns
        -------
        Numeric1DArray
            PDF values for the provided points.

        Raises
        ------
        Exception
            If the KDE model is not fitted.
        ValueError
            If the shape of the provided points is not compatible with
            the fitted KDE model.

        """
        if self.error_convolution:
            # ignores the rest of parameters as it only
            # works with the data and its alwas leave1out
            return self._pdf_with_error_convolution()

        if not self._is_fitted():
            raise Exception("Model not fitted. Try excecuting fit function first.")
        eval_points = np.asarray(eval_points)
        obs, dims = eval_points.shape
        if dims != self._d:
            raise ValueError("Eval points must have same dims as data.")
        if obs < 1:
            raise ValueError("Eval points cannot be empty")

        if self._n_eff <= 0:
            return np.zeros(obs)

        pdf = np.zeros(obs)
        kernels = self._kernels[self._eff_mask]
        weights = self._weights[self._eff_mask]
        n = self._n_eff

        # put weights and normalization toghether in each step
        # pdf(point) = sum(ki(point)*wi/(sum(w)-wi))

        if leave1out:
            norm_weights = weights / (n - weights)
            for i, k in enumerate(kernels):
                applied_k = k.pdf(eval_points) * norm_weights[i]
                applied_k[i] = 0
                pdf += applied_k
        else:
            norm_weights = weights / n
            for i, k in enumerate(kernels):
                applied_k = k.pdf(eval_points) * norm_weights[i]
                pdf += applied_k
        if obs == 1:
            # return as float value
            return pdf[0]
        return pdf

    def plot(
        self,
        gr: int = 50,
        figsize: Tuple[int, int] = (8, 6),
        cols: Optional[str] = None,
        **kwargs,
    ):
        """Plot the KDE model applied in a grid.

        Creates a pairplot of the KDE model applied
        in a grid that spans between the data max and min
        values for each dimension.

        Parameters
        ----------
        gr : int, optional
            Grid resolution, number of bins to be taken into
            account for each dimension, by default 50. Note that
            data dimensions and grid resolution determine how many
            points are evaluated, as ``eval_points=gr**dims``. A high
            ``gr`` value can result in a long computation time.
        figsize : Tuple[int, int], optional
            Figure size, by default (8, 6)
        cols : Optional[str], optional
            Column names to plot over the axes, by default None.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with the pairplot.

        Raises
        ------
        Exception
            If the KDE model is not fitted yet.

        """
        if not self._is_fitted():
            raise Exception("Model not fitted. Try excecuting fit function first.")
        # prepare data to plot
        n = gr
        linspaces = tuple(np.linspace(self._mins, self._maxs, num=n).T)
        grid = np.meshgrid(*linspaces, indexing="ij")
        data = np.vstack(tuple(map(np.ravel, grid))).T
        density = self.pdf(eval_points=data)
        data_and_density = np.vstack((data.T, density))
        grids = [axis.reshape(*tuple([n] * self._d)) for axis in data_and_density]
        density_grid = grids[-1]
        if cols is None:
            cols = [f"var{d+1}" for d in range(self._d)]

        fig, axes = plt.subplots(self._d, self._d, figsize=figsize)
        # fig.subplots_adjust(hspace=.5, wspace=.001)
        for i in range(self._d):
            for j in range(self._d):
                if i == j:
                    x = linspaces[i]
                    axis_to_sum = tuple(set(list(np.arange(self._d))) - {i})
                    if len(axis_to_sum):
                        y = density_grid.sum(axis=axis_to_sum)
                    y = y / gr
                    univariate_density_plot(x=x, y=y, ax=axes[i, i])
                else:
                    x, y = np.meshgrid(linspaces[j], linspaces[i])
                    axis_to_sum = tuple(set(list(np.arange(self._d))) - {i} - {j})
                    if len(axis_to_sum):
                        z = density_grid.sum(axis=axis_to_sum)
                    else:
                        z = density_grid
                    if i > j:
                        z = z.T
                    z = z / gr
                    _, im = bivariate_density_plot(
                        x=x, y=y, z=z, colorbar=False, ax=axes[i, j], **kwargs
                    )

        y_axes_to_join = [set([]) for i in range(self._d)]
        for i in range(self._d):
            for j in range(self._d):
                if i != j:
                    # share x axis with univariate density of same column
                    axes[i, j].sharex(axes[j, j])
                    axes[i, j].axis("tight")
                    if j == 0 or (j == 1 and i == 0):
                        for k in range(self._d):
                            if k != j and k != i:
                                y_axes_to_join[i].add(axes[i, j])
                                y_axes_to_join[i].add(axes[i, k])

        for i in range(self._d):
            yatj = list(y_axes_to_join[i])
            if len(yatj) > 1:
                yatj[0].get_shared_y_axes().join(yatj[0], *yatj[1:])

        for i in range(self._d):
            axes[-1, i].set_xlabel(cols[i])
            axes[i, 0].set_ylabel(cols[i])

        fig.colorbar(im, ax=axes.ravel().tolist())
        return fig, axes
