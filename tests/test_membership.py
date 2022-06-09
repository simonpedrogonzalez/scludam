from scludam.membership import DBME
import pytest
import numpy as np

from scludam.utils import Colnames

# from scludam.synthetic import case2_sample0c, case2_sample1c, case2_sample2c
from sklearn.metrics import matthews_corrcoef
from scludam.utils import one_hot_encode
from scludam.synthetic import (
    StarField,
    StarCluster,
    Synthetic,
    BivariateUniform,
    UniformFrustum,
    polar_to_cartesian,
)
from scipy.stats import multivariate_normal


@pytest.fixture
def sample0c():
    n = 1000

    field = StarField(
        pm=BivariateUniform(locs=(-7, 6), scales=(2.5, 2.5)),
        space=UniformFrustum(locs=(118, -31, 1.2), scales=(6, 6, 0.9)),
        n_stars=int(n),
    )
    return Synthetic(star_field=field, clusters=[]).rvs()[
        ["pmra", "pmdec", "p_pm_field"]
    ]


@pytest.fixture
def sample1c(fmix=0.9):
    n = 1000
    n_clusters = 1
    cmix = (1 - fmix) / n_clusters
    flocs = polar_to_cartesian((118, -31, 1.2))

    field = StarField(
        pm=BivariateUniform(locs=(-7, 6), scales=(2.5, 2.5)),
        space=UniformFrustum(locs=(118, -31, 1.2), scales=(6, 6, 0.9)),
        n_stars=int(n * fmix),
    )
    clusters = [
        StarCluster(
            space=multivariate_normal(mean=polar_to_cartesian([121, -28, 1.6]), cov=50),
            pm=multivariate_normal(mean=(-5.75, 7.25), cov=1.0 / 34),
            n_stars=int(n * cmix),
        ),
    ]
    return Synthetic(star_field=field, clusters=clusters).rvs()


@pytest.fixture
def sample2c(fmix=0.6):
    n = 1000
    n_clusters = 2
    cmix = (1 - fmix) / n_clusters
    flocs = polar_to_cartesian((118, -31, 1.2))

    f_end_point_ra = polar_to_cartesian((124, -31, 1.2))
    f_end_point_dec = polar_to_cartesian((118, -25, 1.2))
    f_end_point_plx = polar_to_cartesian((118, -31, 2))

    field = StarField(
        pm=BivariateUniform(locs=(-7, 6), scales=(2.5, 2.5)),
        space=UniformFrustum(locs=(118, -31, 1.2), scales=(6, 6, 0.9)),
        n_stars=int(n * fmix),
    )
    clusters = [
        StarCluster(
            space=multivariate_normal(
                mean=polar_to_cartesian([120.5, -27.25, 1.57]), cov=50
            ),
            pm=multivariate_normal(mean=(-5.4, 6.75), cov=1.0 / 34),
            n_stars=int(n * cmix),
        ),
        StarCluster(
            space=multivariate_normal(
                mean=polar_to_cartesian([121.75, -28.75, 1.63]), cov=50
            ),
            pm=multivariate_normal(mean=(-6.25, 7.75), cov=1.0 / 34),
            n_stars=int(n * cmix),
        ),
    ]
    return Synthetic(star_field=field, clusters=clusters).rvs()


def get_labels(proba):
    if len(proba.shape) == 1:
        return np.ones_like(proba) * -1
    return np.argmax(proba, axis=1) - 1


def mean_sqdiff(proba1, proba2):
    return np.mean((proba1 - proba2) ** 2)


def test_sample0c(sample0c):
    data = sample0c[["pmra", "pmdec"]].values
    proba = sample0c["p_pm_field"].values
    labels = get_labels(proba)
    init_proba = one_hot_encode(labels)
    dbme = DBME().fit(data=data, init_proba=init_proba)
    assert np.allclose(dbme.posteriors, proba)


def test_sample1c(sample1c):
    data = sample1c[["pmra", "pmdec"]].values
    proba = sample1c["p_pm_cluster1"].values
    labels = get_labels(sample1c[["p_pm_field", "p_pm_cluster1"]].values)
    init_proba = one_hot_encode(labels)
    dbme = DBME().fit(data=data, init_proba=init_proba)
    assert dbme.posteriors.shape == init_proba.shape
    res = dbme.posteriors[:, 1]
    assert mean_sqdiff(proba, init_proba[:, 1]) > mean_sqdiff(proba, res)
    assert mean_sqdiff(proba, res) < 0.01


def test_sample2c(sample2c):
    data = sample2c[["pmra", "pmdec"]].values
    proba1 = sample2c["p_pm_cluster1"].values
    proba2 = sample2c["p_pm_cluster2"].values
    labels = get_labels(
        sample2c[["p_pm_field", "p_pm_cluster1", "p_pm_cluster2"]].values
    )
    init_proba = one_hot_encode(labels)
    dbme = DBME().fit(data=data, init_proba=init_proba)
    assert dbme.posteriors.shape == init_proba.shape
    res1 = dbme.posteriors[:, 1]
    res2 = dbme.posteriors[:, 2]
    assert mean_sqdiff(proba1, init_proba[:, 1]) > mean_sqdiff(proba1, res1)
    assert mean_sqdiff(proba1, res1) < 0.01
    assert mean_sqdiff(proba2, init_proba[:, 2]) > mean_sqdiff(proba2, res2)
    assert mean_sqdiff(proba2, res2) < 0.01


# test get posteriors

# test update mixture
