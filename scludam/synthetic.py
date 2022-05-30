import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from typing_extensions import TypedDict
from typing import Optional, Tuple, List, Union, Callable
from attr import attrib, attrs, validators
import copy
import math
from astropy.coordinates import Distance, SkyCoord
from scipy.spatial import ConvexHull
import astropy.units as u

# Helper functions
def is_inside_circle(center, radius, data):
    dx = np.abs(data[:, 0] - center[0])
    dy = np.abs(data[:, 1] - center[1])
    return (
        (dx < radius)
        & (dy < radius)
        & ((dx + dy <= radius) | (dx**2 + dy**2 <= radius**2))
    )


def is_inside_sphere(center, radius, data):
    dx = np.abs(data[:, 0] - center[0])
    dy = np.abs(data[:, 1] - center[1])
    dz = np.abs(data[:, 2] - center[2])
    return dx**2 + dy**2 + dz**2 <= radius**2


# Coordinate transformation
def cartesian_to_polar(coords: Union[np.ndarray, list]):
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


def polar_to_cartesian(coords: Union[np.ndarray, list]):
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
def in_range(min_value, max_value):
    def range_validator(instance, attribute, value):
        if value < float(min_value):
            raise ValueError(f"{attribute.name} attribute must be >= than {min_value}")
        if value > float(max_value):
            raise ValueError(f"{attribute.name} attribute must be <= than {max_value}")

    return range_validator


def dist_has_n_dimensions(n: int):
    def dist_has_n_dimensions_validator(instance, attribute, value):
        if not value.dim:
            raise TypeError(f"{attribute.name} attribute does not have dim property")
        elif value.dim != n:
            raise ValueError(
                f"{attribute.name} attribute must have {n} dimensions, but has {value.dim} dimensions"
            )

    return dist_has_n_dimensions_validator


def has_len(length: int):
    def has_len_validator(instance, attribute, value):
        if len(value) != length:
            raise ValueError(
                f"{attribute.name} attribute must have length {length}, but has length {len(value)}"
            )

    return has_len_validator


# Custom distributions

# unused
@attrs(auto_attribs=True, init=False)
class EDSD(stats.rv_continuous):

    w0: float
    wl: float
    wf: float

    def __init__(self, w0: float, wl: float, wf: float, **kwargs):
        super().__init__(**kwargs)
        if not self._argcheck(w0, wl, wf):
            raise ValueError("Incorrect parameters types or values")
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

    def _argcheck(self, wl, w0, wf):
        if not isinstance(wl, (float, int)):
            raise TypeError("wl parameter expected int or float")
        if not isinstance(w0, (float, int)):
            raise TypeError("w0 parameter expected int or float")
        if not isinstance(wf, (float, int)):
            raise TypeError("wf parameter expected int or float")
        if not (w0 < wf):
            raise ValueError("w0 must be < than wf")
        if self.a:
            if self.b and (self.a > self.b) or np.isclose(self.a, self.b):
                raise ValueError("a must be < than b")
            if np.isclose(self.a, wf):
                raise ValueError("a must be < than wf")
        return True

    def rvs(self, size: int = 1):
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


@attrs(auto_attribs=True)
class UniformSphere(stats._multivariate.multi_rv_frozen):
    center: Tuple[float, float, float] = attrib(
        validator=[
            validators.deep_iterable(
                member_validator=validators.instance_of((int, float))
            ),
            has_len(3),
        ],
        default=(0.0, 0.0, 0.0),
    )
    radius: float = attrib(validator=validators.instance_of((float, int)), default=1.0)
    dim: float = attrib(default=3, init=False)

    def rvs(self, size: int = 1):
        phi = stats.uniform().rvs(size) * 2 * np.pi
        cos_theta = stats.uniform(-1, 2).rvs(size)
        theta = np.arccos(cos_theta)
        r = np.cbrt(stats.uniform().rvs(size)) * self.radius
        x = r * np.sin(theta) * np.cos(phi) + self.center[0]
        y = r * np.sin(theta) * np.sin(phi) + self.center[1]
        z = r * np.cos(theta) + self.center[2]
        return np.vstack((x, y, z)).T

    # TODO: test
    def pdf(self, x: list):
        is_inside = is_inside_sphere(self.center, self.radius, x)
        res = np.array(is_inside, dtype=float)
        res[res > 0] = 1.0 / (4.0 / 3.0 * np.pi * self.radius**3)
        return res


@attrs(auto_attribs=True)
class UniformFrustum(stats._multivariate.multi_rv_frozen):
    locs: Tuple[float, float, float]
    scales: Tuple[float, float, float]
    dim: float = attrib(default=3, init=False)

    def is_inside_shape(self, data):
        mask = np.zeros(data.shape[0])
        polar = cartesian_to_polar(data)
        mask[
            (polar[:, 0] > self.locs[0])
            & (polar[:, 0] < self.locs[0] + self.scales[0])
            & (polar[:, 1] > self.locs[1])
            & (polar[:, 1] < self.locs[1] + self.scales[1])
            & (polar[:, 2] > self.locs[2])
            & (polar[:, 2] < self.locs[2] + self.scales[2])
        ] = 1
        return mask.astype(dtype=bool)

    # volume of a sqare based piramidal frustum
    # which base is given by locs
    def volume(self):
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
            ]
        )
        vertices = polar_to_cartesian(vertices)
        hull = ConvexHull(vertices)
        return hull.volume

    def rvs(self, size=1):
        extremes = np.array(
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
                (
                    self.locs[0] + self.scales[0] / 2,
                    self.locs[1] + self.scales[1] / 2,
                    self.locs[2],
                ),
                (
                    self.locs[0] + self.scales[0] / 2,
                    self.locs[1] + self.scales[1] / 2,
                    self.locs[2] + self.scales[2],
                ),
            ]
        )

        extremes = polar_to_cartesian(extremes)

        xmin = extremes[:, 0].min()
        xmax = extremes[:, 0].max()
        ymin = extremes[:, 1].min()
        ymax = extremes[:, 1].max()
        zmin = extremes[:, 2].min()
        zmax = extremes[:, 2].max()

        data = np.array([])
        while data.shape[0] < size:
            x = stats.uniform(loc=xmin, scale=xmax - xmin).rvs(size=size)
            y = stats.uniform(loc=ymin, scale=ymax - ymin).rvs(size=size)
            z = stats.uniform(loc=zmin, scale=zmax - zmin).rvs(size=size)
            current_data = np.array([x, y, z]).T
            current_data = current_data[self.is_inside_shape(current_data)]
            if data.shape[0] == 0:
                data = current_data
            else:
                data = np.concatenate((data, current_data))

        if data.shape[0] > size:
            data = data[:size, :]
        return data

    def pdf(self, data):
        res = np.zeros(data.shape[0])
        res[self.is_inside_shape(data)] = 1
        return res / self.volume()


@attrs(auto_attribs=True)
class UniformCircle(stats._multivariate.multi_rv_frozen):
    center: Tuple[float, float] = attrib(
        validator=[
            validators.deep_iterable(
                member_validator=validators.instance_of((int, float))
            ),
            has_len(2),
        ],
        default=(0.0, 0.0),
    )
    radius: float = attrib(validator=validators.instance_of((float, int)), default=1.0)
    dim: float = attrib(default=2, init=False)

    def rvs(self, size: int = 1):
        theta = stats.uniform().rvs(size=size) * 2 * np.pi
        r = self.radius * stats.uniform().rvs(size=size) ** 0.5
        x = r * np.cos(theta) + self.center[0]
        y = r * np.sin(theta) + self.center[1]
        return np.vstack((x, y)).T


@attrs(auto_attribs=True)
class BivariateUnifom(stats._multivariate.multi_rv_frozen):

    locs: Tuple[float, float]
    scales: Tuple[float, float]
    dim: float = attrib(default=2, init=False)

    def rvs(self, size: int = 1):
        x = stats.uniform(loc=self.locs[0], scale=self.scales[0]).rvs(size=size)
        y = stats.uniform(loc=self.locs[1], scale=self.scales[1]).rvs(size=size)
        return np.vstack((x, y)).T

    def pdf(self, x: np.ndarray):
        pdfx = stats.uniform(loc=self.locs[0], scale=self.scales[0]).pdf(x[:, 0])
        pdfy = stats.uniform(loc=self.locs[1], scale=self.scales[1]).pdf(x[:, 1])
        return pdfx * pdfy


# attrs class for Three dimensional uniform distribution
@attrs(auto_attribs=True)
class TrivariateUniform(stats._multivariate.multi_rv_frozen):
    locs: Tuple[float, float, float] = (0, 0, 0)
    scales: Tuple[float, float, float] = (1, 1, 1)
    dim: float = attrib(default=3, init=False)

    def rvs(self, size: int = 1):
        x = stats.uniform(loc=self.locs[0], scale=self.scales[0]).rvs(size=size)
        y = stats.uniform(loc=self.locs[1], scale=self.scales[1]).rvs(size=size)
        z = stats.uniform(loc=self.locs[2], scale=self.scales[2]).rvs(size=size)
        return np.vstack((x, y, z)).T

    def pdf(self, x: np.ndarray):
        pdfx = stats.uniform(loc=self.locs[0], scale=self.scales[0]).pdf(x[:, 0])
        pdfy = stats.uniform(loc=self.locs[1], scale=self.scales[1]).pdf(x[:, 1])
        pdfz = stats.uniform(loc=self.locs[2], scale=self.scales[2]).pdf(x[:, 2])
        return pdfx * pdfy * pdfz


# Data generators
@attrs(auto_attribs=True)
class Cluster:

    space: stats._multivariate.multi_rv_frozen = attrib(
        validator=[
            validators.instance_of(stats._multivariate.multi_rv_frozen),
            dist_has_n_dimensions(n=3),
        ]
    )
    pm: stats._multivariate.multi_rv_frozen = attrib(
        validator=[
            validators.instance_of(stats._multivariate.multi_rv_frozen),
            dist_has_n_dimensions(n=2),
        ]
    )
    representation_type: str = attrib(
        validator=validators.in_(["cartesian", "spherical"]), default="spherical"
    )
    star_count: int = attrib(
        validator=[validators.instance_of(int), in_range(0, "inf")], default=200
    )

    # TODO: fails when n is 1
    def rvs(self):
        size = self.star_count
        data = pd.DataFrame()
        xyz = np.atleast_2d(self.space.rvs(size))
        if self.representation_type == "spherical":
            data["ra"], data["dec"], data["parallax"] = cartesian_to_polar(xyz).T
        else:
            data[["x", "y", "z"]] = pd.DataFrame(xyz)
        pm = np.atleast_2d(self.pm.rvs(size))
        data[["pmra", "pmdec"]] = pd.DataFrame(pm)
        return data

    # test
    def pdf(self, data):
        pm_pdf = self.pm.pdf(data[["pmra", "pmdec"]].to_numpy())
        if set(["x", "y", "z"]).issubset(set(data.columns)):
            space_pdf = self.space.pdf(data[["x", "y", "z"]].to_numpy())
        else:
            xyz = polar_to_cartesian(data["ra", "dec", "parallax"].to_numpy())
            space_pdf = self.space.pdf(xyz)
        return pm_pdf * space_pdf

    def pmpdf(self, data):
        return self.pm.pdf(data[["pmra", "pmdec"]].to_numpy())

    def spacepdf(self, data):
        if set(["x", "y", "z"]).issubset(set(data.columns)):
            space_pdf = self.space.pdf(data[["x", "y", "z"]].to_numpy())
        else:
            xyz = polar_to_cartesian(data["ra", "dec", "parallax"].to_numpy())
            space_pdf = self.space.pdf(xyz)
        return space_pdf


@attrs(auto_attribs=True)
class Field:
    space: stats._multivariate.multi_rv_frozen = attrib(
        validator=[
            validators.instance_of(stats._multivariate.multi_rv_frozen),
            dist_has_n_dimensions(n=3),
        ]
    )
    pm: stats._multivariate.multi_rv_frozen = attrib(
        validator=[
            validators.instance_of(stats._multivariate.multi_rv_frozen),
            dist_has_n_dimensions(n=2),
        ]
    )
    representation_type: str = attrib(
        validator=validators.in_(["cartesian", "spherical"]), default="spherical"
    )
    star_count: int = attrib(
        validator=[validators.instance_of(int), in_range(0, "inf")], default=int(1e5)
    )

    # TODO: test
    def rvs(self):
        size = self.star_count
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
    def pdf(self, data):
        pm_pdf = self.pm.pdf(data[["pmra", "pmdec"]].to_numpy())
        if set(["x", "y", "z"]).issubset(set(data.columns)):
            space_pdf = self.space.pdf(data[["x", "y", "z"]].to_numpy())
        else:
            xyz = polar_to_cartesian(data["ra", "dec", "parallax"].to_numpy())
            space_pdf = self.space.pdf(xyz)
        return pm_pdf * space_pdf

    def pmpdf(self, data):
        return self.pm.pdf(data[["pmra", "pmdec"]].to_numpy())

    def spacepdf(self, data):
        if set(["x", "y", "z"]).issubset(set(data.columns)):
            space_pdf = self.space.pdf(data[["x", "y", "z"]].to_numpy())
        else:
            xyz = polar_to_cartesian(data["ra", "dec", "parallax"].to_numpy())
            space_pdf = self.space.pdf(xyz)
        return space_pdf


@attrs(auto_attribs=True)
class Synthetic:
    field: Field = attrib(validator=validators.instance_of(Field))
    clusters: List[Cluster] = attrib(
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(Cluster)
        )
    )
    representation_type: str = attrib(
        validator=validators.in_(["cartesian", "spherical"]), default="spherical"
    )

    # TODO: test
    def rvs(self):
        self.field.representation_type = "cartesian"
        data = self.field.rvs()
        for i in range(len(self.clusters)):
            self.clusters[i].representation_type = "cartesian"
            cluster_data = self.clusters[i].rvs()
            data = pd.concat([data, cluster_data], axis=0)

        # TODO: improve
        total_stars = sum([c.star_count for c in self.clusters]) + self.field.star_count
        field_mixing_ratio = float(self.field.star_count) / float(total_stars)
        field_p = self.field.pdf(data) * field_mixing_ratio

        field_pmp = self.field.pmpdf(data) * field_mixing_ratio
        field_spacep = self.field.spacepdf(data) * field_mixing_ratio

        clusters_mixing_ratios = [
            float(c.star_count) / float(total_stars) for c in self.clusters
        ]
        cluster_ps = np.array(
            [
                self.clusters[i].pdf(data) * clusters_mixing_ratios[i]
                for i in range(len(self.clusters))
            ]
        )

        cluster_pmps = np.array(
            [
                self.clusters[i].pmpdf(data) * clusters_mixing_ratios[i]
                for i in range(len(self.clusters))
            ]
        )
        cluster_spaceps = np.array(
            [
                self.clusters[i].spacepdf(data) * clusters_mixing_ratios[i]
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


def one_cluster_sample(field_size=int(1e4)):
    field = Field(
        pm=stats.multivariate_normal(mean=(0.0, 0.0), cov=20),
        space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=10),
        star_count=field_size,
    )
    clusters = [
        Cluster(
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([120.7, -28.5, 5]), cov=0.5
            ),
            pm=stats.multivariate_normal(mean=(0.5, 0), cov=1.0 / 35),
            star_count=200,
        ),
    ]
    df = Synthetic(field=field, clusters=clusters).rvs()
    return df


def three_clusters_sample():
    field_size = int(1e4)
    cluster_size = int(2e2)
    field = Field(
        pm=stats.multivariate_normal(mean=(0.0, 0.0), cov=20),
        space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=20),
        star_count=field_size,
    )
    clusters = [
        Cluster(
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([120.7, -28.5, 5]), cov=0.5
            ),
            pm=stats.multivariate_normal(mean=(0.5, 0), cov=1.0 / 10),
            star_count=cluster_size,
        ),
        Cluster(
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([120.8, -28.6, 5]), cov=0.5
            ),
            pm=stats.multivariate_normal(mean=(4.5, 4), cov=1.0 / 10),
            star_count=cluster_size,
        ),
        Cluster(
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([120.9, -28.7, 5]), cov=0.5
            ),
            pm=stats.multivariate_normal(mean=(7.5, 7), cov=1.0 / 10),
            star_count=cluster_size,
        ),
    ]
    df = Synthetic(field=field, clusters=clusters).rvs()
    return df


def sample3c(fmix=0.9):
    field_size = int(1e4)
    cluster_size = int(2e2)

    field = Field(
        pm=stats.multivariate_normal(mean=(0.0, 0.0), cov=20),
        space=UniformFrustum(locs=(118, -31, 1.2), scales=(6, 6, 1)),
        star_count=field_size,
    )
    clusters = [
        Cluster(
            pm=stats.multivariate_normal(mean=(0.5, 0), cov=1.0 / 10),
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([120, -28, 1.3]), cov=0.7
            ),
            star_count=cluster_size,
        ),
        Cluster(
            pm=stats.multivariate_normal(mean=(4.5, 4), cov=1.0 / 10),
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([121, -27, 1.7]), cov=0.7
            ),
            star_count=cluster_size,
        ),
        Cluster(
            pm=stats.multivariate_normal(mean=(7.5, 7), cov=1.0 / 10),
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([122.5, -25.5, 2.1]), cov=0.7
            ),
            star_count=cluster_size,
        ),
    ]
    return Synthetic(field=field, clusters=clusters).rvs()


""" df = sample3c()
#df_plot = df[['ra', 'dec', 'parallax', 'pmra', 'pmdec']]
#sns.pairplot(df_plot, markers='.')
import os
import sys
sys.path.append(os.path.join(os.path.dirname("scludam"), "."))
from scludam.plots import membership_3d_plot

membership_3d_plot(df[['x', 'y', 'z']].values, df['p_field'].values)

plt.show()
print('coso') """


def one_cluster_sample_small(field_size=int(1e3), cluster_size=int(2e2)):
    field = Field(
        pm=stats.multivariate_normal(mean=(0.0, 0.0), cov=10),
        space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=10),
        star_count=field_size,
    )
    clusters = [
        Cluster(
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([120.7, -28.5, 5]), cov=1.5
            ),
            pm=stats.multivariate_normal(mean=(0.5, 0), cov=1.0 / 35),
            star_count=cluster_size,
        ),
    ]
    df = Synthetic(field=field, clusters=clusters).rvs()
    return df


def case1_sample0c():
    field = Field(
        pm=BivariateUnifom(locs=(-7, 6), scales=(2.5, 2.5)),
        space=UniformSphere(center=polar_to_cartesian((121.25, -28, 1.6)), radius=28),
        star_count=1000,
    )
    return Synthetic(field=field, clusters=[]).rvs()


def case1_sample1c(fmix=0.6):
    n = 1000
    n_clusters = 1
    cmix = (1 - fmix) / n_clusters
    field = Field(
        pm=BivariateUnifom(locs=(-7, 6), scales=(2.5, 2.5)),
        space=UniformSphere(center=polar_to_cartesian((121.25, -28, 1.6)), radius=28),
        star_count=int(n * fmix),
    )
    clusters = [
        Cluster(
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([121, -28, 1.6]), cov=7
            ),
            pm=stats.multivariate_normal(mean=(-5.75, 7.25), cov=1.0 / 34),
            star_count=int(n * cmix),
        ),
    ]
    return Synthetic(field=field, clusters=clusters).rvs()


def case1_sample2c(fmix=0.6):
    n = 1000
    n_clusters = 2
    cmix = (1 - fmix) / n_clusters
    field = Field(
        pm=BivariateUnifom(locs=(-7, 6), scales=(2.5, 2.5)),
        space=UniformSphere(center=polar_to_cartesian((121.25, -28, 1.6)), radius=28),
        star_count=int(n * fmix),
    )
    clusters = [
        Cluster(
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([120.5, -27.25, 1.57]), cov=7
            ),
            pm=stats.multivariate_normal(mean=(-5.4, 6.75), cov=1.0 / 34),
            star_count=int(n * cmix),
        ),
        Cluster(
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([121.75, -28.75, 1.63]), cov=7
            ),
            pm=stats.multivariate_normal(mean=(-6.25, 7.75), cov=1.0 / 34),
            star_count=int(n * cmix),
        ),
    ]
    return Synthetic(field=field, clusters=clusters).rvs()


def case2_sample0c(fmix=None):
    n = 1000

    field = Field(
        pm=BivariateUnifom(locs=(-7, 6), scales=(2.5, 2.5)),
        space=UniformFrustum(locs=(118, -31, 1.2), scales=(6, 6, 0.9)),
        star_count=int(n),
    )
    return Synthetic(field=field, clusters=[]).rvs()


def case2_sample1c(fmix=0.6):
    n = 1000
    n_clusters = 1
    cmix = (1 - fmix) / n_clusters
    flocs = polar_to_cartesian((118, -31, 1.2))

    field = Field(
        pm=BivariateUnifom(locs=(-7, 6), scales=(2.5, 2.5)),
        space=UniformFrustum(locs=(118, -31, 1.2), scales=(6, 6, 0.9)),
        star_count=int(n * fmix),
    )
    clusters = [
        Cluster(
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([121, -28, 1.6]), cov=50
            ),
            pm=stats.multivariate_normal(mean=(-5.75, 7.25), cov=1.0 / 34),
            star_count=int(n * cmix),
        ),
    ]
    return Synthetic(field=field, clusters=clusters).rvs()


def case2_sample2c(fmix=0.6):
    n = 1000
    n_clusters = 2
    cmix = (1 - fmix) / n_clusters
    flocs = polar_to_cartesian((118, -31, 1.2))

    f_end_point_ra = polar_to_cartesian((124, -31, 1.2))
    f_end_point_dec = polar_to_cartesian((118, -25, 1.2))
    f_end_point_plx = polar_to_cartesian((118, -31, 2))

    field = Field(
        pm=BivariateUnifom(locs=(-7, 6), scales=(2.5, 2.5)),
        space=UniformFrustum(locs=(118, -31, 1.2), scales=(6, 6, 0.9)),
        star_count=int(n * fmix),
    )
    clusters = [
        Cluster(
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([120.5, -27.25, 1.57]), cov=50
            ),
            pm=stats.multivariate_normal(mean=(-5.4, 6.75), cov=1.0 / 34),
            star_count=int(n * cmix),
        ),
        Cluster(
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([121.75, -28.75, 1.63]), cov=50
            ),
            pm=stats.multivariate_normal(mean=(-6.25, 7.75), cov=1.0 / 34),
            star_count=int(n * cmix),
        ),
    ]
    return Synthetic(field=field, clusters=clusters).rvs()


def case2_sample2c_big(fmix=0.6):
    n = 1000
    n_clusters = 2
    cmix = (1 - fmix) / n_clusters

    field = Field(
        pm=BivariateUnifom(locs=(-7, 6), scales=(10.5, 10.5)),
        space=UniformFrustum(locs=(118, -31, 1.2), scales=(36, 36, 2)),
        star_count=int(n * fmix),
    )
    clusters = [
        Cluster(
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([120.5, -27.25, 1.57]), cov=50
            ),
            pm=stats.multivariate_normal(mean=(-5.4, 6.75), cov=1.0 / 34),
            star_count=int(n * cmix),
        ),
        Cluster(
            space=stats.multivariate_normal(
                mean=polar_to_cartesian([121.75, -28.75, 1.63]), cov=50
            ),
            pm=stats.multivariate_normal(mean=(-6.25, 7.75), cov=1.0 / 34),
            star_count=int(n * cmix),
        ),
    ]
    return Synthetic(field=field, clusters=clusters).rvs()


# df = case2_sample2c_big()
# print('coso')
# df = case2_sample1c(fmix=.6)
# print('coso')


""" uf = UniformFrustum(locs=(118, -25, 1.6), scales=(1,1,.1))
data = uf.rvs(size=1000)
vol = uf.volume()
fig, ax = plt.subplots(subplot_kw={ 'projection': '3d' })
ax.scatter3D(data[:,0], data[:,1], data[:,2])
plt.show()   """
""" df = case2_sample2c(.9)
df = df[['ra', 'dec', 'pmra', 'pmdec', 'parallax']]
sns.pairplot(df)
plt.show()
print('coso') """
""" rt = 'spherical'
field = Field(
    pm=stats.multivariate_normal(mean=(0., 0.), cov=3),
    # space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=700),
    space=UniformSphere(center=polar_to_cartesian((120.5, -27.5, 5)), radius=3),
    representation_type=rt,
    star_count=int(60)
)
cluster = Cluster(
    space=stats.multivariate_normal(
        mean=polar_to_cartesian([120.7, -28.5, 5]),
        cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ),
    pm=stats.multivariate_normal(mean=(.5, .5), cov=.5),
    representation_type=rt,
    star_count=40
)
synthetic = Synthetic(field=field, clusters=[cluster])
data = synthetic.rvs()
data[['ra', 'dec', 'parallax']] = cartesian_to_polar(data[['x', 'y', 'z']].to_numpy())
sns.scatterplot(data=data, x='ra', y='dec', hue='p_cluster1')
plt.show() """
