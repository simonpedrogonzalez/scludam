import numpy as np
import pytest
from astropy.table.table import Table
from scipy.stats import gaussian_kde
from scipy.stats._multivariate import multi_rv_frozen
from sklearn.datasets import load_iris
from utils import raises_exception

from scludam import HKDE
from scludam.hkde import PluginSelector, RuleOfThumbSelector
from scludam.hkde import r as rsession
from scludam.rutils import disable_r_console_output, disable_r_warnings, load_r_packages
from scludam.utils import Colnames

disable_r_console_output()
disable_r_warnings()

df = Table.read("tests/data/ngc2527_small.xml").to_pandas()
cnames = Colnames(df.columns.to_list())
datanames = cnames.data(["pmra", "pmdec", "parallax"])
errornames = cnames.error()
corrnames = cnames.corr()
data = df[datanames].to_numpy()
err = df[errornames].to_numpy()
corr = df[corrnames].to_numpy()
n, d = data.shape
w = np.ones(n)


@pytest.fixture
def kskde():
    from rpy2.robjects import default_converter, numpy2ri, r

    # from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    load_r_packages(r, ["ks"])
    obs, dims = data.shape
    with localconverter(default_converter + numpy2ri.converter):
        r.assign("data", data)
    r("result <- kde(data, eval.points=data)")
    H = np.asarray(r('result["H"]'))
    pdf = np.asarray(r('result["estimate"]'))

    return pdf.ravel(), H


@pytest.fixture
def pdf_with_error_correct():
    return (
        Table.read("tests/data/ngc2527_small_pdf_with_error.txt", format="ascii")
        .to_pandas()["pdf"]
        .to_numpy()
        .ravel()
    )


class TestBandwidths:
    @pytest.mark.parametrize(
        "nstage, pilot, binned, diag, exception, correct_command, correct_result",
        [
            (
                None,
                None,
                None,
                None,
                None,
                "ks::Hpi(x=x)",
                np.array(
                    [
                        [0.00572097, -0.00057937, -0.00015549],
                        [-0.00057937, 0.00593521, 0.00023931],
                        [-0.00015549, 0.00023931, 0.00061363],
                    ]
                ),
            ),
            (
                1,
                "dscalar",
                True,
                True,
                None,
                "ks::Hpi.diag(x=x,nstage=nstage,pilot=pilot,binned=binned)",
                np.array(
                    [
                        [0.01211393, 0.0, 0.0],
                        [0.0, 0.01140754, 0.0],
                        [0.0, 0.0, 0.0013611],
                    ]
                ),
            ),
            (
                1,
                "dscalar",
                type("SomeUnsuportedType", (object,), {})(),
                False,
                TypeError,
                None,
                None,
            ),
        ],
    )
    def test_plugin_bandwidth__build_r_command_and_returns_correct_result(
        self,
        nstage,
        pilot,
        binned,
        diag,
        exception,
        correct_command,
        correct_result,
    ):
        if exception is None:
            command = raises_exception(
                exception,
                lambda: PluginSelector(
                    nstage=nstage, pilot=pilot, binned=binned, diag=diag
                )._build_r_command(data),
            )
            # verify command
            assert command == correct_command
            # verify vars are correctly set
            params = [
                ("nstage", nstage, "1L"),
                ("pilot", pilot, f'"{pilot}"'),
                ("binned", binned, "TRUE"),
            ]
            for param in params:
                if param[1] is not None:
                    assert rsession(f"{param[0]}").r_repr() == param[2]
            assert np.all(np.isclose(np.asarray(rsession("x")), data))

            bw = PluginSelector(
                nstage=nstage, pilot=pilot, binned=binned, diag=diag
            ).get_bw(data)
            assert np.all(np.isclose(correct_result, bw))

    def test_rule_of_thimb_selector__yields_equal_results_than_scipy(self):
        iris = load_iris()
        data = iris.data

        hkde = HKDE(bw=RuleOfThumbSelector(rule="silverman")).fit(data)
        hkde_res = hkde.pdf(data, leave1out=False)

        scipy_kde = gaussian_kde(data.T, bw_method="silverman")
        scipy_res = scipy_kde.evaluate(data.T)

        # selector
        rots = RuleOfThumbSelector(rule="silverman")
        assert rots._silverman_factor(data) == scipy_kde.silverman_factor()
        assert rots._scotts_factor(data) == scipy_kde.scotts_factor()
        weights = np.ones(data.shape[0])
        assert np.allclose(
            rots._get_data_covariance(data, weights), scipy_kde._data_covariance
        )
        assert np.allclose(scipy_res, hkde_res)

    def test_rule_of_thumb_selector_diag__yields_equal_results_than_scipy(self):
        zero_corr_dataset = np.array([[8, 6, 4, 6, 8], [10, 12, 14, 16, 18]]).T
        scipy_kde = gaussian_kde(zero_corr_dataset.T, bw_method="silverman")
        scipy_cov = scipy_kde.covariance
        rots = RuleOfThumbSelector(rule="silverman", diag=True)
        hkde_cov = HKDE(bw=rots).fit(zero_corr_dataset)._covariances[0]
        assert np.allclose(scipy_cov, hkde_cov)


class TestHKDE:
    @pytest.mark.parametrize(
        "bw, exception, correct",
        [
            (
                PluginSelector(),
                None,
                np.array(
                    [
                        [0.00572097, -0.00057937, -0.00015549],
                        [-0.00057937, 0.00593521, 0.00023931],
                        [-0.00015549, 0.00023931, 0.00061363],
                    ]
                ),
            ),
            (
                np.array([0.1, 0.1, 0.05]),
                None,
                np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.05]]),
            ),
            (np.array([0.1, 0.1, 0.05, 0.1]), ValueError, None),
            (
                np.array([[1, -0.05, -0.05], [-0.05, 1, -0.05], [-0.05, -0.05, 1]]),
                None,
                np.array([[1, -0.05, -0.05], [-0.05, 1, -0.05], [-0.05, -0.05, 1]]),
            ),
            (
                np.array([[1, -0.05, -0.05], [-0.05, 1, -0.05]]),
                ValueError,
                None,
            ),
        ],
    )
    def test_get_bw_matrices(self, bw, exception, correct):
        result = raises_exception(
            exception,
            lambda: HKDE(bw=bw, d=d, n=n).set_weights(w)._get_bw_matrices(data),
        )
        if exception is None:
            assert result.shape == (n, d, d)
            assert np.allclose(result[0], correct)

    @pytest.mark.parametrize(
        "corr_param, exception, correct",
        [
            (
                corr,
                None,
                np.array(
                    [
                        [1.0, 0.14773652, 0.14375634],
                        [0.14773652, 1.0, -0.08130455],
                        [0.14375634, -0.08130455, 1.0],
                    ]
                ),
            ),
            (
                np.array([[1, -0.05, -0.05], [-0.05, 1, 25], [-0.05, -0.05, 1]]),
                None,
                np.array([[1, -0.05, -0.05], [-0.05, 1, -0.05], [-0.05, -0.05, 1]]),
            ),
            (np.array([[1, -0.05, -0.05], [-0.05, 1, 25]]), ValueError, None),
            (
                np.array([[1, -0.05, -0.05], [-0.05, 2, 25], [-0.05, -0.05, 1]]),
                ValueError,
                None,
            ),
        ],
    )
    def test_get_corr_matrices(self, corr_param, exception, correct):
        result = raises_exception(
            exception, lambda: HKDE(d=d, n=n)._get_corr_matrices(corr_param)
        )
        if exception is None:
            assert result.shape == (n, d, d)
            assert np.allclose(result[0], correct)

    @pytest.mark.parametrize(
        "err_param, corr_param, exception, correct",
        [
            (
                err,
                None,
                None,
                np.array(
                    [
                        [0.00018201, 0.0, 0.0],
                        [0.0, 0.0002134, 0.0],
                        [0.0, 0.0, 0.00025714],
                    ]
                ),
            ),
            (
                err,
                corr,
                None,
                np.array(
                    [
                        [1.82014118e-04, 2.91164708e-05, 3.11002520e-05],
                        [2.91164708e-05, 2.13401104e-04, -1.90457259e-05],
                        [3.11002520e-05, -1.90457259e-05, 2.57139440e-04],
                    ]
                ),
            ),
            # TODO: test when beartyping in place (None, corr, ValueError, None),
            (
                np.array([[1, -0.05, -0.05], [-0.05, 1, -0.05], [-0.05, -0.05, 1]]),
                None,
                ValueError,
                None,
            ),
        ],
    )
    def test_get_err_matrices(self, err_param, corr_param, exception, correct):
        result = raises_exception(
            exception,
            lambda: HKDE(d=d, n=n)._get_err_matrices(err_param, corr_param),
        )
        if exception is None:
            assert result.shape == (n, d, d)
            assert np.allclose(result[0], correct)

    @pytest.mark.parametrize(
        "err_param, corr_param, correct",
        [
            (None, None, np.diag(np.ones(d) * 2)),
            (
                err,
                corr,
                np.array(
                    [
                        [1.82014118e-04, 2.91164708e-05, 3.11002520e-05],
                        [2.91164708e-05, 2.13401104e-04, -1.90457259e-05],
                        [3.11002520e-05, -1.90457259e-05, 2.57139440e-04],
                    ]
                )
                + np.diag(np.ones(d) * 2),
            ),
        ],
    )
    def test_get_cov_matrices(self, err_param, corr_param, correct):
        # no error
        result = HKDE(bw=np.diag(np.ones(d) * 2), d=d, n=n)._get_cov_matrices(
            data, err_param, corr_param
        )
        assert result.shape == (n, d, d)
        assert np.allclose(result[0], correct)

    @pytest.mark.parametrize(
        "weights, exception, n_eff, eff_mask",
        [
            (np.ones(n) * 0.5, None, n / 2, np.array([True] * n)),
            (
                np.concatenate((np.ones(300) * 0, np.ones(n - 300)), 0),
                None,
                n - 300,
                np.array([False] * 300 + [True] * (n - 300)),
            ),
            (np.ones(n) * 1e-8, None, 0, np.array([False] * n)),
            (np.ones(n - 1), ValueError, None, None),
            (np.ones(n) * 2, ValueError, None, None),
            (np.ones((n, 2)), ValueError, None, None),
        ],
    )
    def test_set_weights(self, weights, exception, n_eff, eff_mask):
        result = raises_exception(exception, lambda: HKDE(n=n).set_weights(weights))
        if exception is None:
            assert isinstance(result, HKDE)
            assert result._n_eff == n_eff
            assert np.allclose(result._eff_mask, eff_mask)
            assert np.allclose(weights, result._weights)

    def test_fit(self):
        hkde = HKDE().fit(data=data, err=err, corr=corr, weights=w)
        assert hkde._n == n
        assert hkde._d == d
        assert np.allclose(hkde._weights, w)
        assert hkde._kernels.shape == (n,)
        assert isinstance(hkde._kernels[0], multi_rv_frozen)
        assert hkde._covariances.shape == (n, d, d)
        assert hkde._is_fitted()

    def test_basic_pdf(self, kskde):
        pdf, H = kskde
        hkde = HKDE().fit(data=data)
        result = hkde.pdf(data, leave1out=False)
        assert result.shape == (n,)
        assert np.allclose(hkde._covariances[0], H)
        assert np.allclose(result, pdf)

    def test_pdf_with_error(self, pdf_with_error_correct):
        result = HKDE().fit(data=data, err=err, corr=corr).pdf(data, leave1out=False)
        assert result.shape == (n,)
        assert np.allclose(result, pdf_with_error_correct)


@pytest.mark.mpl_image_compare
def test_hkde_plot():
    return HKDE().fit(load_iris().data).plot(gr=20)[0]


@pytest.mark.mpl_image_compare
def test_hkde_plot_2d():
    data = load_iris().data[:, :2]
    return HKDE().fit(data).plot(gr=20)[0]
