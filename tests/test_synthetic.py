import math

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import ConvexHull
from scipy.stats import kstest, multivariate_normal

from scludam.synthetic import (
    EDSD,
    BivariateUniform,
    StarCluster,
    StarField,
    Synthetic,
    TrivariateUniform,
    UniformCircle,
    UniformFrustum,
    UniformSphere,
    cartesian_to_polar,
    is_inside_circle,
    is_inside_sphere,
    polar_to_cartesian,
)


# TODO: change for test_if_raises_exception
class Ok:
    pass


# Could be done in other more reliable way
# for example, by comparing cdfs
def assert_is_uniform(x, iters=1, acc=0.9):
    assert (
        np.array(
            [
                kstest(x, "uniform", N=x.size, args=(x.min(), x.max() - x.min())).pvalue
                > 0.05
                for i in range(iters)
            ]
        )
        .astype(int)
        .sum()
        / iters
        > acc
    )


def verify_result(test, func):
    if issubclass(test, Exception):
        with pytest.raises(test):
            func()
    else:
        func()


class TestEDSD:
    @pytest.mark.parametrize(
        "w0, wl, wf, a, b, n, test",
        [
            (1, 2, 3, None, None, 100, Ok),
            (1, 2, 3, 0, 5, 100, Ok),
            (1.1, 4.1, -0.15, None, None, 100, ValueError),
            (1, 2, 3, 5, 3, 100, ValueError),
            (1, 2, 3, 4, 5, 100, ValueError),
        ],
    )
    def test_attrs(self, w0, wl, wf, a, b, n, test):
        verify_result(test, lambda: EDSD(w0=w0, wl=wl, wf=wf, a=a, b=b).rvs(n))

    def test_EDSD_rvs(self):
        sample = EDSD(wl=1.1, w0=-0.15, wf=4.1).rvs(size=100)
        assert sample.dtype == "float64"
        assert sample.shape == (100,)
        assert sample.min() >= -0.15
        assert sample.max() <= 4.1
        sample = EDSD(a=0, b=3, wl=1.1, w0=-0.15, wf=4.1).rvs(size=100)
        assert sample.min() >= max(0, -0.15)
        assert sample.max() <= min(3, 4.1)


class TestHelpers:
    def test_coord_transform(self):
        cartesian = np.random.uniform(low=-16204.0, high=16204.0, size=(1000, 3))
        polar = cartesian_to_polar(cartesian)
        assert np.allclose(cartesian, polar_to_cartesian(polar))

    def test_coord_transform_one_element(self):
        cartesian = np.random.uniform(low=-16204.0, high=16204.0, size=(1, 3)).flatten()
        polar = cartesian_to_polar(cartesian).flatten()
        assert np.allclose(cartesian, polar_to_cartesian(polar))


class TestHelperDistributions:
    def test_uniform_circle_rvs(self):
        center = np.random.uniform(size=2)
        radius = np.random.uniform()
        size = int(1e2)
        assert UniformCircle().dim == 2
        data = UniformCircle(center=center, radius=radius).rvs(size)
        dx = np.abs(data[:, 0] - center[0])
        dy = np.abs(data[:, 1] - center[1])
        assert data.shape == (size, 2)
        assert data[np.sqrt(dx**2 + dy**2) > radius].shape[0] == 0
        assert data[~is_inside_circle(center, radius, data)].shape[0] == 0
        k = radius / math.sqrt(2)
        square = data[(dx <= k) & (dy <= k)]
        sx, sy = square[:, 0], square[:, 1]
        assert kstest(sx, "uniform", args=(sx.min(), sx.max() - sx.min())).pvalue > 0.05
        assert kstest(sy, "uniform", args=(sy.min(), sy.max() - sy.min())).pvalue > 0.05

    @pytest.mark.parametrize(
        "center, radius, data, res",
        [
            (
                (0, 0),
                1,
                np.array([(0, 0), (0, 1.1)]),
                np.array([1.0 / (np.pi * 1**2), 0.0]),
            ),
            (
                (0, 0),
                2,
                np.array([(0, 0), (0, 2.1)]),
                np.array([1.0 / (np.pi * 2**2), 0.0]),
            ),
        ],
    )
    def test_uniform_circle_pdf(self, center, radius, data, res):
        assert np.allclose(UniformCircle(center, radius).pdf(data), res)

    def test_uniform_sphere_rvs(self):
        center = np.random.uniform(size=3)
        radius = np.random.uniform()
        size = int(1e2)
        assert UniformSphere().dim == 3
        data = UniformSphere(center, radius).rvs(size)
        assert data.shape == (size, 3)
        dx = np.abs(data[:, 0] - center[0])
        dy = np.abs(data[:, 1] - center[1])
        dz = np.abs(data[:, 2] - center[2])
        assert data[np.sqrt(dx**2 + dy**2 + dz**2) > radius].shape[0] == 0
        assert data[~is_inside_sphere(center, radius, data)].shape[0] == 0
        k = radius / math.sqrt(3)
        cube = data[(dx <= k) & (dy <= k) & (dz <= k)]
        sx, sy, sz = cube[:, 0], cube[:, 1], cube[:, 2]
        assert_is_uniform(sx)
        assert_is_uniform(sy)
        assert_is_uniform(sz)

    @pytest.mark.parametrize(
        "center, radius, data, res",
        [
            (
                (0, 0, 0),
                1,
                np.array([(0, 0, 0), (0, 0, 1.1)]),
                np.array([1.0 / (4.0 / 3.0 * np.pi * 1**3), 0.0]),
            ),
            (
                (0, 0, 0),
                2,
                np.array([(0, 0, 0), (0, 0, 2.1)]),
                np.array([1.0 / (4.0 / 3.0 * np.pi * 2**3), 0.0]),
            ),
        ],
    )
    def test_uniform_sphere_pdf(self, center, radius, data, res):
        assert np.allclose(UniformSphere(center, radius).pdf(data), res)

    def test_bivariate_uniform_rvs(self):
        size = int(1e2)
        assert BivariateUniform().dim == 2
        data = BivariateUniform().rvs(size)
        assert data.shape == (size, 2)
        assert_is_uniform(data[:, 0])
        assert_is_uniform(data[:, 1])

    @pytest.mark.parametrize(
        "locs, scales, data, res",
        [
            ((0, 0), (1, 1), np.array([(0, 0), (0, 1.1)]), np.array([1.0, 0.0])),
            ((0, 0), (2, 2), np.array([(0, 0), (0, 2.1)]), np.array([1.0 / 4.0, 0.0])),
        ],
    )
    def test_bivariate_uniform_pdf(self, locs, scales, data, res):
        assert np.allclose(BivariateUniform(locs, scales).pdf(data), res)

    def test_trivariate_uniform_rvs(self):
        size = int(1e2)
        assert TrivariateUniform().dim == 3
        data = TrivariateUniform().rvs(size)
        assert data.shape == (size, 3)
        assert_is_uniform(data[:, 0])
        assert_is_uniform(data[:, 1])
        assert_is_uniform(data[:, 2])

    @pytest.mark.parametrize(
        "locs, scales, data, res",
        [
            (
                (0, 0, 0),
                (1, 1, 1),
                np.array([(0, 0, 0), (0, 0, 1.1)]),
                np.array([1.0, 0.0]),
            ),
            (
                (0, 0, 0),
                (2, 2, 2),
                np.array([(0, 0, 0), (0, 0, 2.1)]),
                np.array([1.0 / 8.0, 0.0]),
            ),
        ],
    )
    def test_trivariate_uniform_pdf(self, locs, scales, data, res):
        assert np.allclose(TrivariateUniform(locs, scales).pdf(data), res)

    def test_uniform_frustum_rvs(self):
        locs = np.array([120, -80, 5])
        scales = np.array([1, 2, 2])
        size = int(1e2)
        with pytest.raises(ValueError, match="Invalid ICRS polar coordinate for locs."):
            UniformFrustum((0, 0, 0), (1, 1, 1)).rvs()
        data = UniformFrustum(locs, scales).rvs(size)
        assert data.shape == (size, 3)
        polar = cartesian_to_polar(data)
        assert np.all(polar.min(axis=0) >= locs)
        assert np.all(polar.max(axis=0) <= locs + scales)
        # uniformity is not tested because rvs is
        # generating and discarding points from
        # uniform distributions.

    @pytest.mark.parametrize(
        "locs, scales, data",
        [
            (
                (120, -80, 5),
                (1, 1, -1),
                np.array([(120.1, -79.5, 4.5), (122, -79.5, 4.5)]),
            )
        ],
    )
    def test_uniform_frustum_pdf(self, locs, scales, data):
        vertices = [
            (120, -80, 5),
            (121, -80, 5),
            (120, -79, 5),
            (121, -79, 5),
            (120, -80, 4),
            (121, -80, 4),
            (120, -79, 4),
            (121, -79, 4),
        ]
        inside = 1.0 / ConvexHull(polar_to_cartesian(np.array(vertices))).volume
        assert np.allclose(
            UniformFrustum(locs, scales).pdf(polar_to_cartesian(data)),
            np.array([inside, 0]),
        )


# TODO: update
class TestStarField:
    @pytest.mark.parametrize(
        "space, pm, n, rt, test",
        [
            (UniformSphere(), multivariate_normal((0, 0)), 1, "cartesian", Ok),
            (
                UniformCircle(),
                multivariate_normal((0, 0)),
                1,
                "cartesian",
                ValueError,
            ),
            (
                multivariate_normal((0, 0, 0)),
                multivariate_normal((0, 0)),
                100,
                "spherical",
                Ok,
            ),
            (
                UniformSphere(),
                multivariate_normal(),
                1,
                "cartesian",
                ValueError,
            ),
            (
                UniformSphere(),
                multivariate_normal((0, 0)),
                -1,
                "cartesian",
                ValueError,
            ),
            (UniformSphere(), multivariate_normal((0, 0)), 1, "spherical", Ok),
            (
                UniformSphere(),
                multivariate_normal((0, 0)),
                1,
                "other",
                ValueError,
            ),
        ],
    )
    def test_attrs(self, space, pm, n, rt, test):
        verify_result(
            test,
            lambda: StarField(
                space=space, pm=pm, representation_type=rt, n_stars=n
            ).rvs(),
        )

    def test_rvs(self):
        df = StarField(
            space=UniformSphere(),
            pm=multivariate_normal((0, 0)),
            n_stars=int(1e5),
            representation_type="spherical",
        ).rvs()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (int(1e5), 5)
        assert sorted(list(df.columns)) == sorted(
            ["ra", "dec", "parallax", "pmra", "pmdec"]
        )
        df = StarField(
            space=UniformSphere(),
            pm=multivariate_normal((0, 0)),
            n_stars=int(1e5),
            representation_type="cartesian",
        ).rvs()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (int(1e5), 5)
        assert sorted(list(df.columns)) == sorted(["x", "y", "z", "pmra", "pmdec"])


# TODO: update
class TestStarCluster:
    @pytest.mark.parametrize(
        "space, pm, n, rt, test",
        [
            (UniformSphere(), multivariate_normal((0, 0)), 1, "cartesian", Ok),
            (
                UniformCircle(),
                multivariate_normal((0, 0)),
                1,
                "cartesian",
                ValueError,
            ),
            (
                multivariate_normal((0, 0, 0)),
                multivariate_normal((0, 0)),
                100,
                "spherical",
                Ok,
            ),
            (
                UniformSphere(),
                multivariate_normal(),
                1,
                "cartesian",
                ValueError,
            ),
            (
                UniformSphere(),
                multivariate_normal((0, 0)),
                -1,
                "cartesian",
                ValueError,
            ),
            (UniformSphere(), multivariate_normal((0, 0)), 1, "spherical", Ok),
            (
                UniformSphere(),
                multivariate_normal((0, 0)),
                1,
                "other",
                ValueError,
            ),
        ],
    )
    def test_attrs(self, space, pm, n, rt, test):
        verify_result(
            test,
            lambda: StarCluster(
                space=space, pm=pm, representation_type=rt, n_stars=n
            ).rvs(),
        )

    def test_rvs(self):
        cluster_data = StarCluster(
            space=multivariate_normal((0, 0, 0)),
            pm=multivariate_normal((0, 0)),
            n_stars=100,
            representation_type="spherical",
        ).rvs()
        assert isinstance(cluster_data, pd.DataFrame)
        assert cluster_data.shape == (100, 5)
        assert sorted(list(cluster_data.columns)) == sorted(
            ["ra", "dec", "parallax", "pmra", "pmdec"]
        )
        cluster_data = StarCluster(
            space=multivariate_normal((0, 0, 0)),
            pm=multivariate_normal((0, 0)),
            n_stars=100,
            representation_type="cartesian",
        ).rvs()
        assert isinstance(cluster_data, pd.DataFrame)
        assert cluster_data.shape == (100, 5)
        assert sorted(list(cluster_data.columns)) == sorted(
            ["x", "y", "z", "pmra", "pmdec"]
        )


# TODO
class TestSynthetic:
    @pytest.mark.parametrize(
        "field, clusters, rt, test",
        [
            (
                StarField(
                    space=UniformSphere(),
                    pm=multivariate_normal((0, 0)),
                    n_stars=10,
                ),
                [
                    StarCluster(
                        space=UniformSphere(),
                        pm=multivariate_normal((0, 0)),
                        n_stars=10,
                    )
                ],
                "cartesian",
                Ok,
            ),
            (
                StarField(
                    space=UniformSphere(),
                    pm=multivariate_normal((0, 0)),
                    n_stars=10,
                ),
                [
                    StarCluster(
                        space=UniformSphere(),
                        pm=multivariate_normal((0, 0)),
                        n_stars=10,
                    )
                ],
                "spherical",
                Ok,
            ),
            (
                StarField(
                    space=UniformSphere(),
                    pm=multivariate_normal((0, 0)),
                    n_stars=10,
                ),
                [
                    StarCluster(
                        space=UniformSphere(),
                        pm=multivariate_normal((0, 0)),
                        n_stars=10,
                    )
                ],
                "other",
                ValueError,
            ),
        ],
    )
    def test_attrs(self, field, clusters, rt, test):
        verify_result(
            test,
            lambda: Synthetic(
                star_field=field, clusters=clusters, representation_type=rt
            ).rvs(),
        )

    def test_rvs(self):
        s = Synthetic(
            clusters=[
                StarCluster(
                    space=multivariate_normal((0, 0, 0)),
                    pm=multivariate_normal((0, 0)),
                    n_stars=100,
                ),
                StarCluster(
                    space=multivariate_normal((0.5, 0.5, 0.5)),
                    pm=multivariate_normal((0.5, 0.5)),
                    n_stars=50,
                ),
            ],
            star_field=StarField(
                space=UniformSphere(radius=10),
                pm=multivariate_normal((0, 0)),
                n_stars=int(1e5),
            ),
        )
        s.representation_type = "cartesian"
        df = s.rvs()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (int(1e5) + 100 + 50, 14)
        assert sorted(list(df.columns)) == sorted(
            [
                "x",
                "y",
                "z",
                "pmra",
                "pmdec",
                "p_cluster1",
                "p_cluster2",
                "p_field",
                "p_pm_cluster1",
                "p_pm_cluster2",
                "p_pm_field",
                "p_space_cluster1",
                "p_space_cluster2",
                "p_space_field",
            ]
        )

        s.representation_type = "spherical"
        df = s.rvs()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (int(1e5) + 100 + 50, 15)
        assert sorted(list(df.columns)) == sorted(
            [
                "ra",
                "dec",
                "parallax",
                "log10_parallax",
                "pmra",
                "pmdec",
                "p_cluster1",
                "p_cluster2",
                "p_field",
                "p_pm_cluster1",
                "p_pm_cluster2",
                "p_pm_field",
                "p_space_cluster1",
                "p_space_cluster2",
                "p_space_field",
            ]
        )
