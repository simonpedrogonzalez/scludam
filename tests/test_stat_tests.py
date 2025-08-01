import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal
from sklearn.datasets import load_iris

from scludam import DipDistTest, HopkinsTest, RipleysKTest
from scludam.synthetic import BivariateUniform


@pytest.fixture
def iris():
    return load_iris().data


@pytest.fixture
def uniform_sample():
    return BivariateUniform(locs=(0, 0), scales=(1, 1)).rvs(1000)


@pytest.fixture
def one_cluster_sample():
    sample = BivariateUniform(locs=(0, 0), scales=(1, 1)).rvs(500)
    sample2 = multivariate_normal(mean=(0.5, 0.5), cov=1.0 / 200).rvs(500)
    return np.concatenate((sample, sample2))


@pytest.fixture
def two_clusters_sample():
    sample = BivariateUniform(locs=(0, 0), scales=(1, 1)).rvs(500)
    sample2 = multivariate_normal(mean=(0.75, 0.75), cov=1.0 / 200).rvs(250)
    sample3 = multivariate_normal(mean=(0.25, 0.25), cov=1.0 / 200).rvs(250)
    return np.concatenate((sample, sample2, sample3))


def test_hopkins_uniform_not_rejectH0(uniform_sample):
    assert (
        not HopkinsTest(metric="mahalanobis", n_iters=100)
        .test(data=uniform_sample)
        .rejectH0
    )


def test_hopkins_one_cluster_rejectH0(one_cluster_sample):
    assert (
        HopkinsTest(metric="mahalanobis", n_iters=100)
        .test(data=one_cluster_sample)
        .rejectH0
    )


def test_hopkins_two_clusters_rejectH0(two_clusters_sample):
    assert (
        HopkinsTest(metric="mahalanobis", n_iters=100)
        .test(data=two_clusters_sample)
        .rejectH0
    )


def test_hopkins_iris_rejectH0(iris):
    """Compare Hopkins implementation with R hopkins https://kwstat.github.io/hopkins/.

    hopkins(X, m=150, U=U)
    X is sklearn iris rejectH0 though rpy2
    m is number of samples, taken as all iris, so random sampling does not
    affect the result. U is uniform 4-variate distribution created with numpy
    uniform from seed 0, with locs and scales given by sklearn iris
    value: 0.9978868058086875
    pvalue: 0.0

    """
    ht = HopkinsTest(n_iters=1, sample_ratio=1, metric="euclidean").test(iris)
    assert ht.rejectH0
    assert np.isclose(0.9978868058086875, ht.value, atol=1e-2)
    assert np.isclose(0.0, ht.pvalue, atol=1e-2)


def test_dip_uniform_not_rejectH0(uniform_sample):
    assert not DipDistTest().test(uniform_sample).rejectH0


def test_dip_one_cluster_rejectH0(one_cluster_sample):
    assert DipDistTest().test(one_cluster_sample).rejectH0


def test_dip_two_clusters_rejectH0(two_clusters_sample):
    assert DipDistTest().test(data=two_clusters_sample).rejectH0


def test_dip_iris_rejectH0(iris):
    assert DipDistTest().test(iris).rejectH0


@pytest.mark.parametrize(
    "mode, pvalue, rejectH0",
    [
        ("ripley", 0.05, True),
        ("ripley", 0.01, False),
        ("chiu", 0.1, True),
        ("chiu", 0.05, True),
        ("chiu", 0.01, False),
    ],
)
def test_ripleysk_empirical_rule(mode, pvalue, rejectH0):
    radii = np.array([1, 2, 3])
    lf = np.array([1.01, 2.01, 3.02])
    area = 1.5
    n = 100
    value1, result = RipleysKTest(
        pvalue_threshold=pvalue, mode=mode
    )._empirical_csr_rule(radii=radii, l_function=lf, area=area, n=n)
    assert result == rejectH0
    assert np.isclose(value1, 0.02)


def test_ripley_factor():
    radii = np.array([1, 2, 3])
    lf = np.array([1.01, 2.01, 3.02])
    area = 1.5
    n = 100
    value1, rejectH0 = RipleysKTest(factor=1.65)._empirical_csr_rule(
        radii=radii, l_function=lf, area=area, n=n
    )
    assert not rejectH0
    assert np.isclose(value1, 0.02)


def test_ripleysk_uniform_not_rejectH0(uniform_sample):
    assert not RipleysKTest(mode="ks").test(data=uniform_sample).rejectH0
    assert (
        not RipleysKTest(mode="chiu", pvalue_threshold=0.01)
        .test(data=uniform_sample)
        .rejectH0
    )


def test_ripleysk_one_cluster_rejectH0(one_cluster_sample):
    assert RipleysKTest().test(data=one_cluster_sample).rejectH0
    assert RipleysKTest(mode="ks").test(data=one_cluster_sample).rejectH0


def test_ripleysk_two_cluster_rejectH0(two_clusters_sample):
    assert RipleysKTest(mode="ks").test(data=two_clusters_sample).rejectH0
    assert RipleysKTest().test(data=two_clusters_sample).rejectH0


def test_ripleysk_iris_rejectH0_and_delete_repeated_values(iris):
    spatstat_result = pd.read_csv("tests/data/spatstat_iris.csv")
    spatstat_value = 0.06850111556141059
    with pytest.warns(UserWarning) as record:
        rk = RipleysKTest(pvalue_threshold=0.01, max_samples=5000).test(iris[:, :2])
        assert rk.rejectH0
        assert np.allclose(rk.radii, spatstat_result.radius.to_numpy())
        # difference due to astropy implementation
        assert np.allclose(
            rk.l_function, spatstat_result.l_function.to_numpy(), atol=1e-2
        )
        assert np.isclose(spatstat_value, rk.value, atol=1e-2)
    message = "There are repeated data points that cause "
    message += "astropy.stats.RipleysKEstimator to fail, they will be removed."
    assert record[0].message.args[0] == message
