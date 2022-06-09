import numpy as np
import pytest
from astropy.table.table import Table
from scipy.stats._multivariate import multi_rv_frozen
from utils import raises_exception
from scludam.hkde import HKDE, PluginSelector, r as rsession
from scludam.rutils import load_r_packages, disable_r_console_output, disable_r_warnings
from scludam.utils import Colnames

disable_r_console_output()
disable_r_warnings()

df = Table.read("tests/data/ngc2527_small.xml").to_pandas()
cnames = Colnames(df.columns.to_list())
datanames = cnames.get_data_names(["pmra", "pmdec", "parallax"])
errornames, _ = cnames.get_error_names()
corrnames, _ = cnames.get_corr_names()
data = df[datanames].to_numpy()
err = df[errornames].to_numpy()
corr = df[corrnames].to_numpy()
n, d = data.shape
w = np.ones(n)


@pytest.fixture
def kskde():
    from rpy2.robjects import numpy2ri, r
    from rpy2.robjects import default_converter, conversion
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
    def test_plugin_bandwidth(
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
                ).build_r_command(data),
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
            ).get_bandwidth(data)
            assert np.all(np.isclose(correct_result, bw))


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
            lambda: HKDE(bw=bw, d=d, n=n).set_weights(w).get_bw_matrices(data),
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
            exception, lambda: HKDE(d=d, n=n).get_corr_matrices(corr_param)
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
            lambda: HKDE(d=d, n=n).get_err_matrices(err_param, corr_param),
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
        result = HKDE(bw=np.diag(np.ones(d) * 2), d=d, n=n).get_cov_matrices(
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
            assert result.n_eff == n_eff
            assert np.allclose(result.eff_mask, eff_mask)
            assert np.allclose(weights, result.weights)

    def test_fit(self):
        hkde = HKDE().fit(data=data, err=err, corr=corr, weights=w)
        assert hkde.n == n
        assert hkde.d == d
        assert np.allclose(hkde.weights, w)
        assert hkde.kernels.shape == (n,)
        assert isinstance(hkde.kernels[0], multi_rv_frozen)
        assert hkde.covariances.shape == (n, d, d)
        assert hkde.check_is_fitted()

    def test_basic_pdf(self, kskde):
        pdf, H = kskde
        hkde = HKDE().fit(data=data)
        result = hkde.pdf(data, leave1out=False)
        assert result.shape == (n,)
        assert np.allclose(hkde.covariances[0], H)
        assert np.allclose(result, pdf)

    def test_pdf_with_error(self, pdf_with_error_correct):
        result = HKDE().fit(data=data, err=err, corr=corr).pdf(data, leave1out=False)
        assert result.shape == (n,)
        assert np.allclose(result, pdf_with_error_correct)
