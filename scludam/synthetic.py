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

"""Module for synthetic data generation.

Contains some helpful functions and distributions and the main classes for generating
synthetic sky region samples with an star field and star clusters.

"""

from numbers import Number
from typing import List, Union

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Distance, SkyCoord
from attrs import Factory, define, field, validators
from beartype import beartype
from numpy.typing import NDArray
from scipy import stats
from scipy.spatial import ConvexHull

from scludam.type_utils import (
    Numeric1DArray,
    Numeric2DArray,
    Vector2,
    Vector2Array,
    Vector3,
    Vector3Array,
    _type,
)


# Helper functions
@beartype
def is_inside_circle(
    center: Vector2, radius: Number, data: Union[Vector2, Vector2Array]
) -> NDArray[bool]:
    """Check if data is inside a circle.

    Parameters
    ----------
    center : Vector2
        list, touple of numpy array of 2 number elements, representing the center
        of the circle
    radius : Number
        radius of the circle
    data : Union[Vector2, Vector2Array]
        numpy numeric array of shape (n, 2) to check if inside the circle

    Returns
    -------
    NDArray[bool]
        mask indicating if data is inside the circle.

    """
    data = np.array(data)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    dx = np.abs(data[:, 0] - center[0])
    dy = np.abs(data[:, 1] - center[1])
    return (
        (dx <= radius)
        & (dy <= radius)
        & ((dx + dy <= radius) | (dx**2 + dy**2 <= radius**2))
    )


@beartype
def is_inside_sphere(
    center: Vector3, radius: Number, data: Union[Vector3, Vector3Array]
) -> NDArray[bool]:
    """Check if data is inside a sphere.

    Parameters
    ----------
    center : Vector3
        list, touple of numpy array of 3 number elements, representing the center
        of the sphere
    radius : Number
        radius of the sphere
    data : Union[Vector3, Vector3Array]
        numeric array of shape (n, 3) to check if inside the sphere

    Returns
    -------
    NDArray[bool]
        mask indicating if data is inside the sphere.

    """
    data = np.array(data)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    dx = np.abs(data[:, 0] - center[0])
    dy = np.abs(data[:, 1] - center[1])
    dz = np.abs(data[:, 2] - center[2])
    return dx**2 + dy**2 + dz**2 <= radius**2


# Coordinate transformation
@beartype
def cartesian_to_polar(
    coords: Union[Vector3, Vector3Array]
) -> Union[Vector3, Vector3Array]:
    """Convert cartesian coordinates to polar coordinates.

    Cartesian coordinates are taken as (x, y, z) in parsecs in
    ICRS system and are transformed to (ra, dec, parallax).

    Parameters
    ----------
    coords : Union[Vector3, Vector3Array]
        cartesian coordinates in (x, y, z) in parsecs in ICRS system
        to be transformed to (ra, dec, parallax).

    Returns
    -------
    Union[Vector3, Vector3Array]
        polar coordinates in (ra, dec, parallax).

    """
    coords = np.array(coords)
    if len(coords.shape) == 1:
        coords = SkyCoord(
            x=coords[0],
            y=coords[1],
            z=coords[2],
            unit="pc",
            representation_type="cartesian",
            frame="icrs",
        )
        coords.representation_type = "spherical"
        return np.array([coords.ra.deg, coords.dec.deg, coords.distance.parallax.mas])
    else:
        coords = SkyCoord(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            unit="pc",
            representation_type="cartesian",
            frame="icrs",
        )
        coords.representation_type = "spherical"
        return np.vstack(
            (coords.ra.deg, coords.dec.deg, coords.distance.parallax.mas)
        ).T


@beartype
def polar_to_cartesian(
    coords: Union[Vector3, Vector3Array]
) -> Union[Vector3, Vector3Array]:
    """Convert polar coordinates to cartesian coordinates.

    Polar coordinates are taken as (ra, dec, parallax) in (degree, degree, mas)
    in ICRS system and are transformed to (x, y, z) in parsecs in ICRS system.

    Parameters
    ----------
    coords : Union[Vector3, Vector3Array]
        polar coordinates in (ra, dec, parallax) in (degree, degree, mas)
        in ICRS system.

    Returns
    -------
    Union[Vector3, Vector3Array]
        cartesian coordinates in (x, y, z) in parsecs in ICRS system.

    """
    coords = np.array(coords)
    if len(coords.shape) == 1:
        coords = SkyCoord(
            ra=coords[0] * u.degree,
            dec=coords[1] * u.degree,
            distance=Distance(parallax=coords[2] * u.mas),
            representation_type="spherical",
            frame="icrs",
        )
        coords.representation_type = "cartesian"
        return np.array([coords.x.value, coords.y.value, coords.z.value])
    else:
        coords = SkyCoord(
            ra=coords[:, 0] * u.degree,
            dec=coords[:, 1] * u.degree,
            distance=Distance(parallax=coords[:, 2] * u.mas),
            representation_type="spherical",
            frame="icrs",
        )
        coords.representation_type = "cartesian"
        return np.vstack((coords.x.value, coords.y.value, coords.z.value)).T


# Custom validators
def _in_range(min_value, max_value):
    def range_validator(instance, attribute, value):
        if value < float(min_value):
            raise ValueError(f"{attribute.name} attribute must be >= than {min_value}")
        if value > float(max_value):
            raise ValueError(f"{attribute.name} attribute must be <= than {max_value}")

    return range_validator


def _dist_has_n_dimensions(n: int):
    def _dist_has_n_dimensions_validator(instance, attribute, value):
        if value.dim != n:
            raise ValueError(
                f"{attribute.name} attribute must have {n} dimensions,               "
                f" but has {value.dim} dimensions"
            )

    return _dist_has_n_dimensions_validator


# Custom distributions
@define(init=False)
class EDSD(stats.rv_continuous):
    """Class to represent the EDSD distribution.

    Attributes
    ----------
    w0 : float
        Distribution zero point that indicates the lower limit
    wl : float
        Parameter that determines the width and the peak at wl/4
        of the profile
    wf : float
        Distribution final point that indicates the upper limit

    Returns
    -------
    EDSD:
        Distribution object

    Raises
    ------
    ValueError
        If ``wf < w0``, or if a and b, which determine de evaluation
        domain in scipy.rv_continuous, do not verify ``a < w0``, ``b > wf``
        and ``b > a``.

    Notes
    -----
    Exponentially Decreasing Space Density is used to represent
    certain distributions, such as a parallax distribution of a
    star catalogue [1]_ [2]_. The EDSD distribution is defined as:

    *  ``f(w) = wl**3 / 2*(w-w0)**4 * exp(-wl/(w-w0))`` if ``w > w0`` and ``w < wf``
    *  ``f(w) = 0`` if ``w <= w0``
    *  ``f(w) = 0`` if ``w >= wf``

    where:

    *   ``w``: parallax in mas
    *   ``w0``:  distribution zero point that indicates the lower limit
    *   ``wl``: parameter that determines the width and the peak at wl/4
        of the profile
    *   ``wf``: distribution final point that indicates the upper limit
        from which the distribution is zero. This is added to make
        the function domain limited to ``[w0, wf]``, so other
        values outside this range are not evaluated.

    References
    ----------
    .. [1] C. A. L. Bailer-Jones et al. (2018). Estimating Distance from Parallaxes,
        IV, Distances to 1.33 Billion Stars in Gaia Data Release 2. The Astronomical
        Journal, 156:58 (11pp), 2018 August. https://doi.org/10.3847/1538-3881/aacb21

    .. [2] Z. Shao & L. Li (2019). Gaia Parallax of Milky Way
        Globular Clusters: A Solution
        of Mixture Model. https://www.researchgate.net/publication/335233416

    Examples
    --------
    .. literalinclude :: ../../examples/synthetic/edsd.py
        :language: python
        :linenos:
    .. image:: ../../examples/synthetic/edsd.png

    """

    w0: float
    wl: float
    wf: float

    def __init__(self, w0: float, wl: float, wf: float, **kwargs):
        super().__init__(**kwargs)
        self._argcheck(w0, wl, wf)
        self.w0 = w0
        self.wl = wl
        self.wf = wf

    def _pdf(self, x, wl, w0, wf):
        return np.piecewise(
            x,
            [(x <= w0) + (x >= wf)],
            [0, lambda x: wl**3 / (2 * (x - w0)) ** 4 * np.exp(-wl / (x - w0))],
        )

    def _ppf(self, y, wl, w0, wf):
        return (
            (-0.007 * wl * 2 + 0.087 * wl - 0.12)
            + w0
            + (0.17 + 0.076 * wl) * y**4
            + (2 - 0.037 * wl + 0.0048 * wl**2)
            * wl
            / (((np.log10(-5 / 4 * np.log10(1.0 - y)) - 3) ** 2) - 6)
        )

    def _cdf(self, x, wl, w0, wf):
        return np.piecewise(
            x,
            [x <= w0, x >= wf],
            [
                0,
                1,
                lambda x: (
                    (
                        2 * x**2
                        + (2 * wl - 4 * w0) * x
                        + 2 * w0**2
                        - 2 * wl * w0
                        + wl**2
                    )
                    * np.exp(-wl / (x - w0))
                    / (2 * (x - w0) ** 2)
                ),
            ],
        )

    def pdf(self, x: Union[Number, Numeric1DArray]) -> Numeric1DArray:
        """Probability density function of the EDSD distribution.

        Parameters
        ----------
        x : Union[Number, Numeric1DArray]
            Data to be evaluated.

        Returns
        -------
        Numeric1DArray
            PDF values.

        Notes
        -----
        The PDF is defined as the density profile given in [1]_ [2]_, but it is
        not used for random generation. Instead, a Percent Point Function
        approximation is used.

        """
        return self._pdf(x, self.wl, self.w0, self.wf)

    def cdf(self, x: Union[Number, Numeric1DArray]) -> Numeric1DArray:
        """Cumulative distribution function.

        Parameters
        ----------
        x : Numeric1DArray
        Returns
        -------
        Numeric1DArray
            Cumulative distribution function.

        Notes
        -----
        The CDF is a polinomial approximation of the real CDF
        function.

        """
        return self._cdf(x, self.wl, self.w0, self.wf)

    def ppf(self, y: Union[Number, Numeric1DArray]) -> Numeric1DArray:
        """Percent point function.

        Parameters
        ----------
        y : Numeric1DArray

        Returns
        -------
        Numeric1DArray
            Percent point function.

        Notes
        -----
        The PPF is a polinomial approximation of the real PPF. As cdf and
        ppf are approximations, one is close to the inverse of the other,
        but not exactly.

        """
        return self._ppf(y, self.wl, self.w0, self.wf)

    @beartype
    def _argcheck(self, wl: Number, w0: Number, wf: Number):
        if not (w0 < wf):
            raise ValueError("w0 must be < than wf")
        if self.a:
            if self.b and (self.a > self.b) or np.isclose(self.a, self.b):
                raise ValueError("a must be < than b")
            if self.a > wf or np.isclose(self.a, wf):
                raise ValueError("a must be < than wf")
        return True

    def rvs(self, size: int = 1) -> Numeric1DArray:
        """Generate random variates from the EDSD distribution.

        Parameters
        ----------
        size : int, optional
            Size of the sample, by default 1

        Returns
        -------
        Numeric1DArray
            Generated sample.

        """
        wl, w0, wf = self.wl, self.w0, self.wf
        self._argcheck(wl, w0, wf)
        limits = np.array([max(w0 + 1e-10, self.a), min(wf - 1e-10, self.b)]).astype(
            "float64"
        )
        rv_limits = self._cdf(limits, wl, w0, wf)
        sample = np.array([])
        while sample.shape[0] < size:
            y = np.random.uniform(low=rv_limits[0], high=rv_limits[1], size=size)
            new_sample = self._ppf(y, wl, w0, wf)
            new_sample = new_sample[
                (new_sample >= limits[0]) & (new_sample <= limits[1])
            ]
            sample = np.concatenate((sample, new_sample), axis=0)
        return sample[:size]


@define
class UniformSphere(stats._multivariate.multi_rv_frozen):
    """Class to represent the Uniform Sphere distribution.

    Attributes
    ----------
    center : Vector3
        Center of the sphere.
    radius : Number
        Radius of the sphere.

    """

    center: Vector3 = field(validator=_type(Vector3), default=(0.0, 0.0, 0.0))
    radius: Number = field(validator=_type(Number), default=1.0)
    dim: int = field(default=3, init=False)

    @beartype
    def rvs(self, size: int = 1) -> Vector3Array:
        """Generate random sample from the Uniform Sphere distribution.

        Parameters
        ----------
        size : int, optional
            Number of samples to be generated, by default 1

        Returns
        -------
        Vector3Array :
            Numpy numeric array of shape (size, 3) with the samples.

        """
        phi = stats.uniform().rvs(size) * 2 * np.pi
        cos_theta = stats.uniform(-1, 2).rvs(size)
        theta = np.arccos(cos_theta)
        r = np.cbrt(stats.uniform().rvs(size)) * self.radius
        x = r * np.sin(theta) * np.cos(phi) + self.center[0]
        y = r * np.sin(theta) * np.sin(phi) + self.center[1]
        z = r * np.cos(theta) + self.center[2]
        return np.vstack((x, y, z)).T

    @beartype
    def pdf(self, x: Union[Vector3, Vector3Array]) -> Numeric1DArray:
        """Probability density function of the Uniform Sphere distribution.

        Is calculated as 0 if the point is outside the sphere,
        and 1 divided by the volume of the sphere otherwise.

        Parameters
        ----------
        x : Union[Vector3, Vector3Array]
            Data to be evaluated

        Returns
        -------
        Numeric1DArray
            Numpy numeric array of shape (size,) with the pdf values.

        """
        is_inside = is_inside_sphere(self.center, self.radius, x)
        res = np.array(is_inside, dtype=float)
        res[res > 0] = 1.0 / (4.0 / 3.0 * np.pi * self.radius**3)
        return res


@define
class UniformFrustum(stats._multivariate.multi_rv_frozen):
    """Class to represent the Uniform Frustum distribution.

    It was defined to represent a square sky region sample. It
    represents a uniform distribution inside a pyramidal frustum.

    Attributes
    ----------
    locs : Vector3
        Reference corner of the frustum. It is given by (ra, dec, parallax)
        polar coordinates in ICRS system, in (degree, degree, mas).
    scales : Vector3
        Size of the frustum in (ra, dec, parallax) polar coordinates in ICRS,
        in (degree, degree, mas).
    max_size_per_iter: int, optional
        Maximum number of samples to be generated per iteration, by default
        1e7. A bigger value can reduce the amount of time needed to
        generate the sample, but take more memory.

    Examples
    --------
    .. literalinclude :: ../../examples/synthetic/uniform_frustum.py
        :language: python
        :linenos:
    .. image:: ../../examples/synthetic/uniform_frustum.png

    """

    locs: Vector3 = field(validator=_type(Vector3))
    scales: Vector3 = field(validator=_type(Vector3))
    dim: int = field(default=3, init=False)
    max_size_per_iter: int = field(default=int(1e7), validator=_type(int))

    def _get_vertices(self):
        vertices = np.array(
            [
                self.locs,
                (self.locs[0] + self.scales[0], self.locs[1], self.locs[2]),
                (self.locs[0], self.locs[1] + self.scales[1], self.locs[2]),
                (
                    self.locs[0] + self.scales[0],
                    self.locs[1] + self.scales[1],
                    self.locs[2],
                ),
                (self.locs[0], self.locs[1], self.locs[2] + self.scales[2]),
                (
                    self.locs[0] + self.scales[0],
                    self.locs[1],
                    self.locs[2] + self.scales[2],
                ),
                (
                    self.locs[0],
                    self.locs[1] + self.scales[1],
                    self.locs[2] + self.scales[2],
                ),
                (
                    self.locs[0] + self.scales[0],
                    self.locs[1] + self.scales[1],
                    self.locs[2] + self.scales[2],
                ),
                # (
                #     self.locs[0] + self.scales[0] / 2,
                #     self.locs[1] + self.scales[1] / 2,
                #     self.locs[2],
                # ),
                # (
                #     self.locs[0] + self.scales[0] / 2,
                #     self.locs[1] + self.scales[1] / 2,
                #     self.locs[2] + self.scales[2],
                # ),
            ]
        )
        vertices = polar_to_cartesian(vertices)
        if not np.isfinite(vertices).all():
            raise ValueError("Invalid ICRS polar coordinate for locs.")
        return vertices

    def _is_inside_shape(self, data: Numeric2DArray):
        mask = np.zeros(data.shape[0])
        polar = cartesian_to_polar(data)
        vertices = cartesian_to_polar(self._get_vertices())

        ramax, decmax, plxmax = tuple(vertices.max(axis=0))
        ramin, decmin, plxmin = tuple(vertices.min(axis=0))

        mask[
            (polar[:, 0] > ramin)
            & (polar[:, 0] < ramax)
            & (polar[:, 1] > decmin)
            & (polar[:, 1] < decmax)
            & (polar[:, 2] > plxmin)
            & (polar[:, 2] < plxmax)
        ] = 1
        return mask.astype(dtype=bool)

    # volume of a sqare based piramidal frustum
    # which base is given by locs. TODO: improve.
    def _volume(self):
        vertices = self._get_vertices()
        hull = ConvexHull(vertices)
        return hull.volume

    @beartype
    def rvs(self, size: int = 1) -> Vector3Array:
        """Generate random sample from the Uniform Frustum distribution.

        Parameters
        ----------
        size : int, optional
            size of the random sample, by default 1

        Returns
        -------
        Vector3Array :
            numpy numeric array of shape (size, 3) with the samples.

        Raises
        ------
        ValueError
            If the given locs are not a valid ICRS coordinate.

        """
        extremes = self._get_vertices()

        xmax, ymax, zmax = tuple(extremes.max(axis=0))
        xmin, ymin, zmin = tuple(extremes.min(axis=0))

        data = np.array([])

        size_per_iter = min(
            int((ymax - ymin) * (xmax - xmin) * (zmax - zmin) / self._volume() * size),
            self.max_size_per_iter,
        )

        while data.shape[0] < size:
            x = stats.uniform(loc=xmin, scale=xmax - xmin).rvs(size=size_per_iter)
            y = stats.uniform(loc=ymin, scale=ymax - ymin).rvs(size=size_per_iter)
            z = stats.uniform(loc=zmin, scale=zmax - zmin).rvs(size=size_per_iter)
            current_data = np.array([x, y, z]).T
            current_data = current_data[self._is_inside_shape(current_data)]
            if data.shape[0] == 0:
                data = current_data
            else:
                data = np.concatenate((data, current_data))

        if data.shape[0] > size:
            data = data[:size, :]
        return data

    @beartype
    def pdf(self, data: Union[Vector3, Vector3Array]) -> Numeric1DArray:
        """Probability density function of the Uniform Frustum distribution.

        It is defined as 1 / volume of the frustum for the points inside the frustum,
        and 0 for the points outside.

        Parameters
        ----------
        data : Union[Vector3, Vector3Array]
            numpy numeric array to be evaluated.

        Returns
        -------
        Numeric1DArray :
            PFD values.

        Raises
        ------
        ValueError
            If the given locs are not a valid ICRS coordinate.

        """
        res = np.zeros(data.shape[0])
        res[self._is_inside_shape(data)] = 1
        return res / self._volume()


@define
class UniformCircle(stats._multivariate.multi_rv_frozen):
    """Uniform Circle distribution.

    Attributes
    ----------
    center : Vector2
        center of the circle.
    radius : Number
        radius of the circle.

    """

    center: Vector2 = field(validator=_type(Vector2), default=(0.0, 0.0))
    radius: Number = field(validator=_type(Number), default=1.0)
    dim: int = field(default=2, init=False)

    @beartype
    def pdf(self, x: Union[Vector2, Vector2Array]) -> Numeric1DArray:
        """Probability density function of the Uniform Circle distribution.

        It is defined as 1 / (pi * radius^2) for the points inside the circle,
        and 0 for the points outside.

        Parameters
        ----------
        x :  Union[Vector2, Vector2Array]
            numpy numeric array to be evaluated.

        Returns
        -------
        Numeric1DArray :
            Numpy numeric array of shape with the pdf values.

        """
        is_inside = is_inside_circle(self.center, self.radius, x)
        res = np.array(is_inside, dtype=float)
        res[res > 0] = 1.0 / (np.pi * self.radius**2)
        return res

    @beartype
    def rvs(self, size: int = 1) -> Vector2Array:
        """Generate random sample from the Uniform Circle distribution.

        Parameters
        ----------
        size : int, optional
            size of the random sample, by default 1

        Returns
        -------
        Vector2Array :
            Numpy numeric array of shape (size, 2) with the samples.

        """
        theta = stats.uniform().rvs(size=size) * 2 * np.pi
        r = self.radius * stats.uniform().rvs(size=size) ** 0.5
        x = r * np.cos(theta) + self.center[0]
        y = r * np.sin(theta) + self.center[1]
        return np.vstack((x, y)).T


@define
class BivariateUniform(stats._multivariate.multi_rv_frozen):
    """Bivariate Uniform distribution.

    Attributes
    ----------
    locs : Vector2
        center of the bivariate uniform distribution.
    scales : Vector2
        scale of the bivariate uniform distribution.

    Returns
    -------
    BivariateUniform :
        instance of the Bivariate Uniform distribution.

    """

    locs: Vector2 = field(validator=_type(Vector2), default=(0.0, 0.0))
    scales: Vector2 = field(validator=_type(Vector2), default=(1.0, 1.0))
    dim: int = field(default=2, init=False)

    @beartype
    def rvs(self, size: int = 1) -> Vector2Array:
        """Generate random sample from the Bivariate Uniform distribution.

        Parameters
        ----------
        size : int, optional
            size of the sample, by default 1

        Returns
        -------
        Vector2Array :
            numpy array of shape (size, 2) with the samples.

        """
        x = stats.uniform(loc=self.locs[0], scale=self.scales[0]).rvs(size=size)
        y = stats.uniform(loc=self.locs[1], scale=self.scales[1]).rvs(size=size)
        return np.vstack((x, y)).T

    @beartype
    def pdf(self, x: Union[Vector2, Vector2Array]) -> Numeric1DArray:
        """Probability density function of the Bivariate Uniform distribution.

        Parameters
        ----------
        x : Union[Vector2, Vector2Array]
            data to be evaluated.

        Returns
        -------
        Numeric1DArray :
            numpy array with the pdf values.

        """
        pdfx = stats.uniform(loc=self.locs[0], scale=self.scales[0]).pdf(x[:, 0])
        pdfy = stats.uniform(loc=self.locs[1], scale=self.scales[1]).pdf(x[:, 1])
        return pdfx * pdfy


@define
class TrivariateUniform(stats._multivariate.multi_rv_frozen):
    """Trivariate Uniform distribution.

    Attributes
    ----------
    locs : Vector3
        center of the trivariate uniform distribution.
    scales : Vector3
        scale of the trivariate uniform distribution.

    Returns
    -------
    TrivariateUniform :
        instance of the Trivariate Uniform distribution.

    """

    locs: Vector3 = field(validator=_type(Vector3), default=(0.0, 0.0, 0.0))
    scales: Vector3 = field(validator=_type(Vector3), default=(1.0, 1.0, 1.0))
    dim: int = field(default=3, init=False)

    @beartype
    def rvs(self, size: int = 1) -> Vector3Array:
        """Generate random sample from the Trivariate Uniform distribution.

        Parameters
        ----------
        size : int, optional
            size of the sample, by default 1

        Returns
        -------
        Vector3Array :
            Numpy array of shape (size, 3) with the samples.

        """
        x = stats.uniform(loc=self.locs[0], scale=self.scales[0]).rvs(size=size)
        y = stats.uniform(loc=self.locs[1], scale=self.scales[1]).rvs(size=size)
        z = stats.uniform(loc=self.locs[2], scale=self.scales[2]).rvs(size=size)
        return np.vstack((x, y, z)).T

    @beartype
    def pdf(self, x: Union[Vector3, Vector3Array]) -> Numeric1DArray:
        """Probability density function of the Trivariate Uniform distribution.

        Is defined as 1 / volume for the points inside the volume, and 0 for
        the points outside.

        Parameters
        ----------
        x : Union[Vector3, Vector3Array]
            Points to be evaluated.

        Returns
        -------
        Numeric1DArray :
            Numpy array with the pdf values.

        """
        pdfx = stats.uniform(loc=self.locs[0], scale=self.scales[0]).pdf(x[:, 0])
        pdfy = stats.uniform(loc=self.locs[1], scale=self.scales[1]).pdf(x[:, 1])
        pdfz = stats.uniform(loc=self.locs[2], scale=self.scales[2]).pdf(x[:, 2])
        return pdfx * pdfy * pdfz


# Data generators
@define
class StarCluster:
    """Class for generating astrometric data from a star cluster.

    Attributes
    ----------
    space : stats._multivariate.multi_rv_frozen
        Space distribution of the cluster. It must have 3 dimensions.
    pm : stats._multivariate.multi_rv_frozen
        Proper motion distribution of the cluster. It must have 3 dimensions.
    representation_type : str
        Type of representation of the cluster. It must be "spherical" or "cartesian",
        by default "spherical".

    Returns
    -------
    StarCluster :
        An instance of the StarCluster class.

    """

    space: stats._multivariate.multi_rv_frozen = field(
        validator=[
            _type(stats._multivariate.multi_rv_frozen),
            _dist_has_n_dimensions(n=3),
        ]
    )
    pm: stats._multivariate.multi_rv_frozen = field(
        validator=[
            _type(stats._multivariate.multi_rv_frozen),
            _dist_has_n_dimensions(n=2),
        ]
    )
    representation_type: str = field(
        validator=[_type(str), validators.in_(["cartesian", "spherical"])],
        default="spherical",
    )
    n_stars: int = field(validator=[_type(int), _in_range(0, "inf")], default=200)

    # TODO: fails when n is 1
    def rvs(self) -> pd.DataFrame:
        """Generate random sample from the Star Cluster distribution.

        Returns
        -------
        pd.DataFrame
            Data frame with the samples.

        """
        size = self.n_stars
        data = pd.DataFrame()
        xyz = np.atleast_2d(self.space.rvs(size))
        if self.representation_type == "spherical":
            data["ra"], data["dec"], data["parallax"] = cartesian_to_polar(xyz).T
        else:
            data[["x", "y", "z"]] = pd.DataFrame(xyz)
        pm = np.atleast_2d(self.pm.rvs(size))
        data[["pmra", "pmdec"]] = pd.DataFrame(pm)
        return data

    def pdf(self, data: pd.DataFrame) -> Numeric1DArray:
        """Joint probability density function of the StarCluster distribution.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be evaluated. Must contain the columns "ra", "dec", "parallax"
            or "x", "y", "z", and "pmra", "pmdec".

        Returns
        -------
        Numeric1DArray
            PDF values for the data.

        """
        pm_pdf = self.pm.pdf(data[["pmra", "pmdec"]].to_numpy())
        if set(["x", "y", "z"]).issubset(set(data.columns)):
            space_pdf = self.space.pdf(data[["x", "y", "z"]].to_numpy())
        else:
            xyz = polar_to_cartesian(data["ra", "dec", "parallax"].to_numpy())
            space_pdf = self.space.pdf(xyz)
        return pm_pdf * space_pdf

    def pm_pdf(self, data: pd.DataFrame) -> Numeric1DArray:
        """Probability density function of the Proper Motion distribution.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be evaluated.

        Returns
        -------
        Numeric1DArray
            PDF values for the data.

        """
        return self.pm.pdf(data[["pmra", "pmdec"]].to_numpy())

    def space_pdf(self, data: pd.DataFrame) -> Numeric1DArray:
        """Probability density function of the space distribution.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be evaluated. Must have columns "x", "y", "z" or
            "ra", "dec", "parallax".

        Returns
        -------
        Numeric1DArray
            PDF values for the data.

        """
        if set(["x", "y", "z"]).issubset(set(data.columns)):
            space_pdf = self.space.pdf(data[["x", "y", "z"]].to_numpy())
        else:
            xyz = polar_to_cartesian(data["ra", "dec", "parallax"].to_numpy())
            space_pdf = self.space.pdf(xyz)
        return space_pdf


@define
class StarField:
    """Class for generating star field data.

    Attributes
    ----------
    space : stats._multivariate.multi_rv_frozen
        Distribution of the space coordinates. Must have 3 dimensions.
    pm : stats._multivariate.multi_rv_frozen
        Distribution of the proper motion coordinates. Must have 2 dimensions.
    representation_type : str
        Type of representation of the spatial data. Must be "cartesian" or "spherical",
        by default "spherical".

    Returns
    -------
    StarField :
        An instance of the StarField class.

    """

    space: stats._multivariate.multi_rv_frozen = field(
        validator=[
            _type(stats._multivariate.multi_rv_frozen),
            _dist_has_n_dimensions(n=3),
        ]
    )
    pm: stats._multivariate.multi_rv_frozen = field(
        validator=[
            _type(stats._multivariate.multi_rv_frozen),
            _dist_has_n_dimensions(n=2),
        ]
    )
    representation_type: str = field(
        validator=[_type(str), validators.in_(["cartesian", "spherical"])],
        default="spherical",
    )
    n_stars: int = field(validator=[_type(int), _in_range(0, "inf")], default=int(1e5))

    # TODO: test
    def rvs(self) -> pd.DataFrame:
        """Generate random sample from the Star Field distribution.

        Returns
        -------
        pd.DataFrame
            Contains columns for the space distrubution and the pm distribution.

        """
        size = self.n_stars
        data = pd.DataFrame()
        xyz = np.atleast_2d(self.space.rvs(size))
        pm = np.atleast_2d(self.pm.rvs(size))
        data[["pmra", "pmdec"]] = pd.DataFrame(np.vstack((pm[:, 0], pm[:, 1])).T)
        if self.representation_type == "spherical":
            ra_dec_plx = cartesian_to_polar(xyz)
            data[["ra", "dec", "parallax"]] = pd.DataFrame(ra_dec_plx)
        else:
            data[["x", "y", "z"]] = pd.DataFrame(xyz)
        return data

    # TODO: test
    def pdf(self, data: pd.DataFrame) -> Numeric1DArray:
        """Joint Probability Density Function of the Star Field.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be evaluated.

        Returns
        -------
        Numeric1DArray
            PDF values for the data.

        """
        pm_pdf = self.pm.pdf(data[["pmra", "pmdec"]].to_numpy())
        if set(["x", "y", "z"]).issubset(set(data.columns)):
            space_pdf = self.space.pdf(data[["x", "y", "z"]].to_numpy())
        else:
            xyz = polar_to_cartesian(data["ra", "dec", "parallax"].to_numpy())
            space_pdf = self.space.pdf(xyz)
        return pm_pdf * space_pdf

    def pm_pdf(self, data: pd.DataFrame) -> Numeric1DArray:
        """Probability density function of the Proper Motion distribution.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be evaluated. Must have columns "pmra" and "pmdec".

        Returns
        -------
        Numeric1DArray
            PDF of the Proper Motion distribution.

        """
        return self.pm.pdf(data[["pmra", "pmdec"]].to_numpy())

    def space_pdf(self, data: pd.DataFrame) -> Numeric1DArray:
        """Probability density function of the space distribution.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be evaluated. Must have columns "x", "y", "z"
            or "ra", "dec", "parallax".

        Returns
        -------
        Numeric1DArray
            PDF of the space distribution.

        """
        if set(["x", "y", "z"]).issubset(set(data.columns)):
            space_pdf = self.space.pdf(data[["x", "y", "z"]].to_numpy())
        else:
            xyz = polar_to_cartesian(data["ra", "dec", "parallax"].to_numpy())
            space_pdf = self.space.pdf(xyz)
        return space_pdf


@define
class Synthetic:
    """Class for generating synthetic data.

    Attributes
    ----------
    space : stats._multivariate.multi_rv_frozen
        The space distribution, it must be a 3d distribution.
    pm : stats._multivariate.multi_rv_frozen
        The proper motion distribution, it must be a 2d distribution.
    representation_type : str
        The representation type, it must be "cartesian" or "spherical",
        by default is "spherical".

    Examples
    --------
    .. literalinclude:: ../../examples/synthetic/synthetic.py
        :language: python
        :linenos:
    .. image:: ../../examples/synthetic/synthetic.png

    """

    star_field: StarField = field(validator=_type(StarField))
    representation_type: str = field(
        validator=[_type(str), validators.in_(["cartesian", "spherical"])],
        default="spherical",
    )
    clusters: List[StarCluster] = Factory(list)

    # TODO: test
    def rvs(self) -> pd.DataFrame:
        """Generate synthetic data from field and cluster distributions.

        Returns
        -------
        pd.DataFrame
            synthetic data with columns for the space distribution, the proper motion
            distribution and approximations of the membership probabilities without
            taking into account any errors.

        """
        self.star_field.representation_type = "cartesian"
        data = self.star_field.rvs()
        for i in range(len(self.clusters)):
            self.clusters[i].representation_type = "cartesian"
            cluster_data = self.clusters[i].rvs()
            data = pd.concat([data, cluster_data], axis=0)

        # TODO: improve
        total_stars = sum([c.n_stars for c in self.clusters]) + self.star_field.n_stars
        field_mixing_ratio = float(self.star_field.n_stars) / float(total_stars)
        field_p = self.star_field.pdf(data) * field_mixing_ratio

        field_pmp = self.star_field.pm_pdf(data) * field_mixing_ratio
        field_spacep = self.star_field.space_pdf(data) * field_mixing_ratio

        clusters_mixing_ratios = [
            float(c.n_stars) / float(total_stars) for c in self.clusters
        ]
        cluster_ps = np.array(
            [
                self.clusters[i].pdf(data) * clusters_mixing_ratios[i]
                for i in range(len(self.clusters))
            ]
        )

        cluster_pmps = np.array(
            [
                self.clusters[i].pm_pdf(data) * clusters_mixing_ratios[i]
                for i in range(len(self.clusters))
            ]
        )
        cluster_spaceps = np.array(
            [
                self.clusters[i].space_pdf(data) * clusters_mixing_ratios[i]
                for i in range(len(self.clusters))
            ]
        )

        total_p = cluster_ps.sum(axis=0) + field_p
        total_pmp = cluster_pmps.sum(axis=0) + field_pmp
        total_spacep = cluster_spaceps.sum(axis=0) + field_spacep

        total_clusters_probs = 0
        total_clusters_pmprobs = 0
        total_clusters_spaceprobs = 0

        for i in range(len(self.clusters)):
            data[f"p_cluster{i+1}"] = cluster_ps[i] / total_p
            total_clusters_probs += cluster_ps[i] / total_p

            data[f"p_pm_cluster{i+1}"] = cluster_pmps[i] / total_pmp
            total_clusters_pmprobs += cluster_pmps[i] / total_pmp

            data[f"p_space_cluster{i+1}"] = cluster_spaceps[i] / total_spacep
            total_clusters_spaceprobs += cluster_spaceps[i] / total_spacep

        data["p_field"] = 1 - total_clusters_probs
        data["p_pm_field"] = 1 - total_clusters_pmprobs
        data["p_space_field"] = 1 - total_clusters_spaceprobs

        if self.representation_type == "spherical":
            xyz = data[["x", "y", "z"]].to_numpy()
            data["ra"], data["dec"], data["parallax"] = cartesian_to_polar(xyz).T
            data["log10_parallax"] = np.log10(data["parallax"])
            data.drop(["x", "y", "z"], inplace=True, axis=1)
        return data
