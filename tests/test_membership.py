import numpy as np
import pytest
from scipy.stats import multivariate_normal

from scludam import HKDE
from scludam.membership import DBME
from scludam.synthetic import (
    BivariateUniform,
    StarCluster,
    StarField,
    Synthetic,
    UniformFrustum,
    polar_to_cartesian,
)
from scludam.utils import one_hot_encode


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


def test_sample0c_returns_0_proba(sample0c):
    data = sample0c[["pmra", "pmdec"]].values
    proba = sample0c["p_pm_field"].values
    labels = get_labels(proba)
    init_proba = one_hot_encode(labels)
    dbme = DBME().fit(data=data, init_proba=init_proba)
    assert dbme._is_fitted()
    assert dbme.n_iters == 2
    assert np.allclose(dbme.posteriors, proba)


def test_sample1c_returns_closer_proba_than_init_proba(sample1c):
    data = sample1c[["pmra", "pmdec"]].values
    proba = sample1c["p_pm_cluster1"].values
    labels = get_labels(sample1c[["p_pm_field", "p_pm_cluster1"]].values)
    init_proba = one_hot_encode(labels)
    dbme = DBME().fit(data=data, init_proba=init_proba)
    assert dbme._is_fitted()
    assert dbme.n_iters == 2
    assert dbme.posteriors.shape == init_proba.shape
    res = dbme.posteriors[:, 1]
    assert mean_sqdiff(proba, init_proba[:, 1]) > mean_sqdiff(proba, res)
    assert mean_sqdiff(proba, res) < 0.01


def test_sample2c_returns_closer_proba_than_init_proba(sample2c):
    data = sample2c[["pmra", "pmdec"]].values
    proba1 = sample2c["p_pm_cluster1"].values
    proba2 = sample2c["p_pm_cluster2"].values
    labels = get_labels(
        sample2c[["p_pm_field", "p_pm_cluster1", "p_pm_cluster2"]].values
    )
    init_proba = one_hot_encode(labels)
    dbme = DBME().fit(data=data, init_proba=init_proba)
    assert dbme._is_fitted()
    assert dbme.n_iters == 2
    assert dbme.posteriors.shape == init_proba.shape
    res1 = dbme.posteriors[:, 1]
    res2 = dbme.posteriors[:, 2]
    assert mean_sqdiff(proba1, init_proba[:, 1]) > mean_sqdiff(proba1, res1)
    assert mean_sqdiff(proba1, res1) < 0.01
    assert mean_sqdiff(proba2, init_proba[:, 2]) > mean_sqdiff(proba2, res2)
    assert mean_sqdiff(proba2, res2) < 0.01


@pytest.fixture
def mock_HKDE_fit(mocker):
    return mocker.patch(
        "scludam.hkde.HKDE.fit",
        return_value=HKDE(),
    )


@pytest.fixture
def mock_HKDE_set_weights(mocker):
    return mocker.patch(
        "scludam.hkde.HKDE.set_weights",
        return_value=HKDE(),
    )


@pytest.fixture
def mock_HKDE_pdf(mocker):
    return mocker.patch(
        "scludam.hkde.HKDE.pdf",
        return_value=np.ones(1000),
    )


def test_corr_err_calls_hkde_with_corr_err(
    sample2c, mock_HKDE_fit, mock_HKDE_set_weights, mock_HKDE_pdf
):
    data = sample2c[["pmra", "pmdec"]].values
    labels = get_labels(
        sample2c[["p_pm_field", "p_pm_cluster1", "p_pm_cluster2"]].values
    )
    init_proba = one_hot_encode(labels)
    dbme = DBME()
    dbme.fit(data=data, init_proba=init_proba, corr=0.5, err=1)
    assert dbme._is_fitted()
    assert mock_HKDE_fit.call_count == 3
    assert mock_HKDE_fit.call_args_list[0][1]["corr"] == 0.5
    assert mock_HKDE_fit.call_args_list[0][1]["err"] == 1

    assert mock_HKDE_fit.call_args_list[1][1]["corr"] == 0.5
    assert mock_HKDE_fit.call_args_list[1][1]["err"] == 1

    assert mock_HKDE_fit.call_args_list[2][1]["corr"] == 0.5
    assert mock_HKDE_fit.call_args_list[2][1]["err"] == 1


def test_same_mode_calls_fit_only_once(
    sample2c, mock_HKDE_fit, mock_HKDE_set_weights, mock_HKDE_pdf
):
    data = sample2c[["pmra", "pmdec"]].values
    labels = get_labels(
        sample2c[["p_pm_field", "p_pm_cluster1", "p_pm_cluster2"]].values
    )
    init_proba = one_hot_encode(labels)
    n_iters = 3
    dbme = DBME(kernel_calculation_mode="same", n_iters=n_iters)
    dbme.fit(data=data, init_proba=init_proba)
    assert dbme._is_fitted()
    assert mock_HKDE_fit.call_count == 1
    assert mock_HKDE_pdf.call_count == n_iters * 3


def test_per_class_mode_calls_fit_once_per_class(
    sample2c, mock_HKDE_fit, mock_HKDE_set_weights, mock_HKDE_pdf
):
    data = sample2c[["pmra", "pmdec"]].values
    labels = get_labels(
        sample2c[["p_pm_field", "p_pm_cluster1", "p_pm_cluster2"]].values
    )
    init_proba = one_hot_encode(labels)
    n_iters = 3
    dbme = DBME(kernel_calculation_mode="per_class", n_iters=n_iters)
    dbme.fit(data=data, init_proba=init_proba)
    assert dbme._is_fitted()
    assert mock_HKDE_fit.call_count == 3
    assert mock_HKDE_pdf.call_count == n_iters * 3


def test_per_class_per_iter_mode_calls_fit_once_per_class_per_iter(
    sample2c, mock_HKDE_fit, mock_HKDE_set_weights, mock_HKDE_pdf
):
    data = sample2c[["pmra", "pmdec"]].values
    labels = get_labels(
        sample2c[["p_pm_field", "p_pm_cluster1", "p_pm_cluster2"]].values
    )
    n_iters = 3
    init_proba = one_hot_encode(labels)
    dbme = DBME(kernel_calculation_mode="per_class_per_iter", n_iters=n_iters)
    dbme.fit(data=data, init_proba=init_proba)
    assert dbme._is_fitted()
    assert mock_HKDE_fit.call_count == n_iters * 3
    assert mock_HKDE_pdf.call_count == n_iters * 3


def test_get_posteriors(sample2c):
    data = sample2c[["pmra", "pmdec"]].values
    labels = get_labels(
        sample2c[["p_pm_field", "p_pm_cluster1", "p_pm_cluster2"]].values
    )
    init_proba = one_hot_encode(labels)
    dbme = DBME().fit(data=data, init_proba=init_proba)
    densities = dbme._get_densities(data, None, None, dbme.posteriors)
    assert np.allclose(dbme._get_posteriors(densities).sum(axis=1), np.ones(len(data)))


def test_multiple_estimator_configuration_errors(sample2c):
    data = sample2c[["pmra", "pmdec"]].values
    labels = get_labels(
        sample2c[["p_pm_field", "p_pm_cluster1", "p_pm_cluster2"]].values
    )
    init_proba = one_hot_encode(labels)

    dbme = DBME(
        # problema es que tenemos 3 clases y 4 estimadores
        pdf_estimator=[HKDE(), HKDE(), HKDE(), HKDE()],
    )
    message = "n_estimators should be 1, 2 or n_classes"
    with pytest.raises(ValueError, match=message):
        dbme.fit(data=data, init_proba=init_proba)


def test_multiple_estimator_configuration_ok(sample2c):
    data = sample2c[["pmra", "pmdec"]].values
    labels = get_labels(
        sample2c[["p_pm_field", "p_pm_cluster1", "p_pm_cluster2"]].values
    )
    init_proba = one_hot_encode(labels)
    bws = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
    dbme = DBME(
        pdf_estimator=[HKDE(bw=bws[0]), HKDE(bw=bws[1]), HKDE(bw=bws[2])],
    )
    dbme.fit(data=data, init_proba=init_proba)
    assert dbme._is_fitted()
    assert len(dbme._estimators) == 3
    for i, estimator in enumerate(dbme._estimators):
        assert isinstance(estimator, HKDE)
        assert np.allclose(estimator.bw, bws[i])
        # que se haya usado el bw correcto
        # la 1er covarianza tenga diagonal igual al bw
        diag = estimator._covariances[init_proba[:, i] > 0][0].diagonal()
        assert np.allclose(diag, bws[i])


def test_two_estimator_configuration_ok(sample2c):
    data = sample2c[["pmra", "pmdec"]].values
    labels = get_labels(
        sample2c[["p_pm_field", "p_pm_cluster1", "p_pm_cluster2"]].values
    )
    init_proba = one_hot_encode(labels)
    bws = [[0.1, 0.1], [0.2, 0.2]]
    dbme = DBME(
        # debería usar 1er para 1era clase y 2ndo para el resto
        pdf_estimator=[HKDE(bw=bws[0]), HKDE(bw=bws[1])],
    )
    dbme.fit(data=data, init_proba=init_proba)
    assert dbme._is_fitted()
    assert len(dbme._estimators) == 3
    for i, estimator in enumerate(dbme._estimators):
        assert isinstance(estimator, HKDE)
        if i == 2:
            j = 1
        else:
            j = i
        assert np.allclose(estimator.bw, bws[j])
        # que se haya usado el bw correcto
        # la 1er covarianza tenga diagonal igual al bw
        diag = estimator._covariances[init_proba[:, i] > 0][0].diagonal()
        assert np.allclose(diag, bws[j])
