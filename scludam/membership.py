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


"""Module for Density Based Membership Estimation."""

from copy import deepcopy

import numpy as np
from attrs import Factory, define, field, validators

from scludam import HKDE
from scludam.plots import (
    pairprobaplot,
    scatter3dprobaplot,
    surfprobaplot,
    tsneprobaplot,
)
from scludam.type_utils import (
    Numeric1DArray,
    Numeric2DArray,
    OptionalNumeric2DArray,
    OptionalNumericArray,
    _type,
)


@define
class DBME:
    """Density Based Membership Estimation.

    It uses :class:`~scludam.hkde.HKDE` to estimate the density and calculate
    smooth membership probabilities for each class, given data and
    initial probabilities.

    Attributes
    ----------
    n_iters : int
        Number of iterations, by default 2. In each iteration,
        prior probabilities are updated according to the posterior
        probabilities of the previous iteration.
    kernel_calculation_mode : str
        Mode of kernel calculation, by default ``per_class``. It indicates how
        many :class:`~scludam.hkde.HKDE` estimators will be used to estimate the
        density.
        Available modes are:

        *  ``same``: the bandwidth of the kernels is the same for all classes. There
           will be one estimator.
        *  ``per_class``: the bandwidth of the kernels is different for each class.
           There will be one estimator per class.
        *  ``per_class_per_iter``: the bandwidth of the kernels is different for
           each class and iteration. There will be one estimator per class which will
           be updated in each iteration, recalculating the bandwith each time.

    kde_leave1out : bool
        Whether to use leave-one-out KDE estimation, by default True.

    pdf_estimator : :class:`~scludam.hkde.HKDE`
        Estimator used to estimate the density, by default an instance of HKDE with
        default parameters.

    n_classes: int
        Number of detected classes. Only available after the
        :func:`~scludam.membership.DBME.fit` method is called.
    labels : Numeric1DArray
        Labels of the classes, only available after the
        :func:`~scludam.membership.DBME.fit` method is called.
    counts : Numeric1DArray
        Number of data points in each class, only available after the
        :func:`~scludam.membership.DBME.fit` method is called.
    priors : Numeric1DArray
        Prior probabilities of each class, only available after the
        :func:`~scludam.membership.DBME.fit` method is called.
    posteriors : Numeric2DArray
        Posterior probabilities array of shape (n_datapoints, n_classes),
        only available after the
        :func:`~scludam.membership.DBME.fit` method is called.

    Examples
    --------
    .. literalinclude:: ../../examples/membership/dbme.py
        :language: python
        :linenos:
    .. image:: ../../examples/membership/init_proba.png
    .. image:: ../../examples/membership/dbme.png
    """

    # intput attrs
    n_iters: int = field(default=2, validator=[_type(int), validators.gt(0)])
    kde_leave1out: bool = field(default=True, validator=_type(bool))
    kernel_calculation_mode: str = field(
        validator=validators.in_(["same", "per_class", "per_class_per_iter"]),
        default="per_class",
    )
    pdf_estimator: HKDE = field(default=HKDE(), validator=_type(HKDE))

    # internal attrs
    _n: int = None
    _d: int = None
    _unique_labels: Numeric1DArray = None
    _data: Numeric2DArray = None
    _estimators: list = Factory(list)
    _iter_priors: list = Factory(list)
    _iter_counts: list = Factory(list)
    _iter_label_diff: list = Factory(list)
    _iter_labels: list = Factory(list)

    # output attrs
    n_classes: int = None
    labels: Numeric1DArray = None
    counts: Numeric1DArray = None
    posteriors: Numeric2DArray = None
    priors: Numeric1DArray = None

    def _update_class_mixtures(self, posteriors: Numeric2DArray):
        self.labels = np.argmax(posteriors, axis=1) - 1
        self._iter_labels.append(self.labels)

        if len(self._iter_labels) > 1:
            label_diff = (
                (self._iter_labels[-1] != self._iter_labels[-2]).astype(int).sum()
            )
            self._iter_label_diff.append(label_diff)

        self.counts = posteriors.sum(axis=0)
        self._iter_counts.append(self.counts)

        self.priors = self.counts / self._n
        self._iter_priors.append(self.priors)

    def _get_posteriors(self, densities: Numeric2DArray):
        # probability calculation
        # P(Ci|x) = Di(x) * P(Ci) / sumj(Dj(x) * P(Cj))
        total_density = (
            (densities * self.counts)
            .sum(axis=1, keepdims=True)
            .repeat(self.n_classes, axis=1)
        )
        posteriors = densities * self.counts / total_density
        return posteriors

    # def get_posteriors2(self, densities):
    #     # probability calculation
    #     # P(Ci|x) = Di(x) / sumj(Dj(x))
    #     total_density = densities.sum(axis=1, keepdims=True).repeat(
    #         self.n_classes, axis=1
    #     )
    #     posteriors = densities / total_density
    #     return posteriors

    # def get_posteriors3(self, densities):
    #     # probability calculation
    #     # P(Ci|x) = Di(x) / sumj(Dj(x))
    #     total_density = densities.sum(axis=1, keepdims=True).repeat(
    #         self.n_classes, axis=1
    #     )
    #     posteriors = densities / total_density

    #     yfie1 = HKDE().fit(self.data, weights=posteriors[:, 0]).pdf(self.data)
    #     yclu = 1 - yfie1
    #     new_posteriors = np.concatenate(
    #         (yfie1.reshape(-1, 1), yclu.reshape(-1, 1)), axis=1
    #     )

    #     return new_posteriors

    def _get_densities(
        self,
        data: Numeric2DArray,
        err: OptionalNumeric2DArray,
        corr: OptionalNumericArray,
        weights: OptionalNumeric2DArray,
    ):
        densities = np.zeros((self._n, self.n_classes))

        # estimator(s) fitting
        if not self._estimators or self.kernel_calculation_mode == "per_class_per_iter":
            if self.kernel_calculation_mode == "same":
                self._estimators = [self.pdf_estimator.fit(data, err, corr)]
            else:
                self._estimators = []
                for i in range(self.n_classes):
                    self._estimators.append(
                        deepcopy(self.pdf_estimator).fit(
                            data=data,
                            err=err,
                            corr=corr,
                            weights=weights[:, i],
                        ),
                    )

        # pdf estimation
        for i in range(self.n_classes):
            if self.kernel_calculation_mode == "same":
                class_estimator = self._estimators[0]
            else:
                class_estimator = self._estimators[i]
            densities[:, i] = class_estimator.set_weights(weights[:, i]).pdf(
                data, leave1out=self.kde_leave1out
            )

        return densities

    def fit(
        self,
        data: Numeric2DArray,
        init_proba: Numeric2DArray,
        err: OptionalNumeric2DArray = None,
        corr: OptionalNumericArray = None,
    ):
        """Fit models and calculate posteriors probabilities.

        The method takes data and initial probabilities and
        creates density estimators. Prior probabilities are
        taken from the initial probabilities. In each iteration,
        the method calculates the posterior probabilities of each
        datapoint using the density estimates and prior probabilites.
        Also, the method updates the prior probabilities considering
        the posterior probabilities of the past iteration.
        ``n_iters=1``
        uses prior probabilities as provided in the initial
        probabilities array. ``n_iters=2`` (recommended),
        updates the prior probabilities once.

        Parameters
        ----------
        data : Numeric2DArray
            Data matrix.
        init_proba : Numeric2DArray
            Initial posterior probability array.
            Must be of shape (n_samples, n_classes). This probabilities
            are used to create the initial density estimators per class.
        err : OptionalNumeric2DArray, optional
            Error parameter to be passed to :func:`~scludam.hkde.HKDE.fit`,
            by default None.
        corr : OptionalNumericArray, optional
            Correlation parameter to be passed to :func:`~scludam.hkde.HKDE.fit`,
            by default None.

        Returns
        -------
        DBME
            Fitted instance of the :class:`~scludam.membership.DBME` class.

        """
        self._n, self._d = data.shape
        self._data = data
        self.labels = np.argmax(init_proba, axis=1) - 1
        self._unique_labels = np.unique(self.labels)
        self.n_classes = len(self._unique_labels)

        self.posteriors = init_proba
        self._update_class_mixtures(posteriors=init_proba)

        # case no clusters found
        if self.n_classes == 1:
            # there are no populations to fit
            return self

        for i in range(self.n_iters):
            # is copy actually needed?
            previous_posteriors = self.posteriors.copy()
            weights = previous_posteriors
            densities = self._get_densities(data, err, corr, weights)
            self.posteriors = self._get_posteriors(densities)
            self._update_class_mixtures(self.posteriors)
        return self

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
            data=self._data, proba=self.posteriors, labels=self.labels, **kwargs
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
            data=self._data, proba=self.posteriors, labels=self.labels, **kwargs
        )

    def _is_fitted(self):
        return (
            self.labels is not None
            and self.posteriors is not None
            and self._data is not None
            and self.priors is not None
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
        return scatter3dprobaplot(self._data, self.posteriors, **kwargs)

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
        return surfprobaplot(self._data, self.posteriors, **kwargs)
