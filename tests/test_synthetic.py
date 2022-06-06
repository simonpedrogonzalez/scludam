import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest, multivariate_normal

from scludam.synthetic import (
    EDSD,
    Cluster,
    Field,
    Synthetic,
    UniformCircle,
    UniformSphere,
    cartesian_to_polar,
    is_inside_circle,
    is_inside_sphere,
    polar_to_cartesian,
)


# TODO: change for test_if_raises_exception
class Ok:
    pass


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

    def test_uniform_circle(self):
        center = np.random.uniform(size=2)
        radius = np.random.uniform()
        size = int(1e7)
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

    def test_uniform_sphere(self):
        center = np.random.uniform(size=3)
        radius = np.random.uniform()
        size = int(1e5)
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
        assert kstest(sx, "uniform", args=(sx.min(), sx.max() - sx.min())).pvalue > 0.05
        assert kstest(sy, "uniform", args=(sy.min(), sy.max() - sy.min())).pvalue > 0.05
        assert kstest(sz, "uniform", args=(sz.min(), sz.max() - sz.min())).pvalue > 0.05


# TODO: update
class TestField:
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
            lambda: Field(
                space=space, pm=pm, representation_type=rt, star_count=n
            ).rvs(),
        )

    def test_rvs(self):
        df = Field(
            space=UniformSphere(),
            pm=multivariate_normal((0, 0)),
            star_count=int(1e5),
            representation_type="spherical",
        ).rvs()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (int(1e5), 5)
        assert sorted(list(df.columns)) == sorted(
            ["ra", "dec", "parallax", "pmra", "pmdec"]
        )
        df = Field(
            space=UniformSphere(),
            pm=multivariate_normal((0, 0)),
            star_count=int(1e5),
            representation_type="cartesian",
        ).rvs()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (int(1e5), 5)
        assert sorted(list(df.columns)) == sorted(["x", "y", "z", "pmra", "pmdec"])


# TODO: update
class TestCluster:
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
            lambda: Cluster(
                space=space, pm=pm, representation_type=rt, star_count=n
            ).rvs(),
        )

    def test_rvs(self):
        cluster_data = Cluster(
            space=multivariate_normal((0, 0, 0)),
            pm=multivariate_normal((0, 0)),
            star_count=100,
            representation_type="spherical",
        ).rvs()
        assert isinstance(cluster_data, pd.DataFrame)
        assert cluster_data.shape == (100, 5)
        assert sorted(list(cluster_data.columns)) == sorted(
            ["ra", "dec", "parallax", "pmra", "pmdec"]
        )
        cluster_data = Cluster(
            space=multivariate_normal((0, 0, 0)),
            pm=multivariate_normal((0, 0)),
            star_count=100,
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
                Field(
                    space=UniformSphere(),
                    pm=multivariate_normal((0, 0)),
                    star_count=10,
                ),
                [
                    Cluster(
                        space=UniformSphere(),
                        pm=multivariate_normal((0, 0)),
                        star_count=10,
                    )
                ],
                "cartesian",
                Ok,
            ),
            (
                Field(
                    space=UniformSphere(),
                    pm=multivariate_normal((0, 0)),
                    star_count=10,
                ),
                [
                    Cluster(
                        space=UniformSphere(),
                        pm=multivariate_normal((0, 0)),
                        star_count=10,
                    )
                ],
                "spherical",
                Ok,
            ),
            (
                Field(
                    space=UniformSphere(),
                    pm=multivariate_normal((0, 0)),
                    star_count=10,
                ),
                [
                    Cluster(
                        space=UniformSphere(),
                        pm=multivariate_normal((0, 0)),
                        star_count=10,
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
                Cluster(
                    space=multivariate_normal((0, 0, 0)),
                    pm=multivariate_normal((0, 0)),
                    star_count=100,
                ),
                Cluster(
                    space=multivariate_normal((0.5, 0.5, 0.5)),
                    pm=multivariate_normal((0.5, 0.5)),
                    star_count=50,
                ),
            ],
            star_field=Field(
                space=UniformSphere(radius=10),
                pm=multivariate_normal((0, 0)),
                star_count=int(1e5),
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
