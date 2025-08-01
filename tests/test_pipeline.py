import numpy as np
from astropy.table.table import Table

from scludam import (
    DBME,
    DEP,
    HKDE,
    SHDBSCAN,
    CountPeakDetector,
    HopkinsTest,
    RipleysKTest,
    RuleOfThumbSelector,
)
from scludam.detection import DetectionResult

sample2cdf = Table.read("tests/data/ngc2323_data.fits").to_pandas()


def test_real_sample2c():
    dep = DEP(
        detector=CountPeakDetector(
            bin_shape=[0.7, 0.7, 0.1],
            min_score=7,
        ),
        estimator=DBME(
            pdf_estimator=HKDE(
                # scotts rule is used insted of plugin
                # because is faster and bw selection
                # is not being tested here
                bw=RuleOfThumbSelector(rule="scott"),
            ),
            # same mode is used because is faster
            # and DBME is not being tested here
            kernel_calculation_mode="same",
        ),
        det_cols=["pmra", "pmdec", "parallax"],
        sample_sigma_factor=2,
        tests=[
            RipleysKTest(max_samples=500),
            HopkinsTest(),
        ],
        test_cols=[["ra", "dec"], ["ra", "dec"]],
        mem_cols=["pmra", "pmdec", "parallax", "ra", "dec"],
    ).fit(sample2cdf)

    assert dep.n_detected == 2
    assert isinstance(dep.detection_result, DetectionResult)
    assert isinstance(dep.detector, CountPeakDetector)
    assert dep.detection_result.scores.size == 2

    assert len(dep.test_results) == 2
    for test_set in dep.test_results:
        assert len(test_set) == 2
        assert test_set[0].rejectH0 or test_set[1].rejectH0

    assert dep.n_estimated == 2
    assert len(dep.estimators) == 2
    assert len(dep.clusterers) == 2
    for est in dep.estimators:
        assert isinstance(est, DBME)
    for clu in dep.clusterers:
        assert isinstance(clu, SHDBSCAN)
    assert np.allclose(dep.estimators[0].counts.round(), np.array([621, 519]))
    assert dep.estimators[0]._data.shape == (1140, 5)
    assert dep.estimators[1]._data.shape == (236, 5)
    assert np.allclose(dep.estimators[1].counts.round(), np.array([185, 51]))

    assert np.allclose(
        dep.proba.sum(axis=0), np.array([15798.8392228, 518.51234219, 50.64843501])
    )
    assert np.allclose(dep.proba.sum(axis=1), 1)


def test_all_test_mode():
    dep = DEP(
        detector=CountPeakDetector(
            bin_shape=[0.7, 0.7, 0.1],
            min_score=7,
        ),
        estimator=DBME(
            pdf_estimator=HKDE(
                bw=RuleOfThumbSelector(rule="scott"),
            ),
            kernel_calculation_mode="same",
        ),
        det_cols=["pmra", "pmdec", "parallax"],
        sample_sigma_factor=2,
        tests=[
            RipleysKTest(max_samples=500),
            HopkinsTest(),
        ],
        test_mode="all",
        test_cols=[["ra", "dec"], ["ra", "dec"]],
    ).fit(sample2cdf)

    assert dep.n_detected == 2
    assert dep.n_estimated == 1


def test_majority_test_mode():
    dep = DEP(
        detector=CountPeakDetector(
            bin_shape=[0.7, 0.7, 0.1],
            min_score=7,
        ),
        estimator=DBME(
            pdf_estimator=HKDE(
                bw=RuleOfThumbSelector(rule="scott"),
            ),
            kernel_calculation_mode="same",
        ),
        det_cols=["pmra", "pmdec", "parallax"],
        sample_sigma_factor=2,
        tests=[
            RipleysKTest(max_samples=500),
            HopkinsTest(),
        ],
        test_mode="majority",
        test_cols=[["ra", "dec"], ["ra", "dec"]],
    ).fit(sample2cdf)

    assert dep.n_detected == 2
    assert dep.n_estimated == 2


def test_default_mem_cols_uses_center_in_clustering():
    dep = DEP(
        detector=CountPeakDetector(
            bin_shape=[0.7, 0.7, 0.1],
            min_score=7,
        ),
        clusterer=SHDBSCAN(
            auto_allow_single_cluster=True,
            min_cluster_size=50,
            noise_proba_mode="conservative",
            cluster_proba_mode="hard",
            # scaler=None so it does not scale the detected centers
            # for clustering and they can be directly compared
            scaler=None,
        ),
        estimator=DBME(
            pdf_estimator=HKDE(
                bw=RuleOfThumbSelector(rule="scott"),
            ),
            kernel_calculation_mode="same",
        ),
        det_cols=["pmra", "pmdec", "parallax"],
    ).fit(sample2cdf)
    assert sorted(dep.mem_cols) == sorted(dep.det_cols)
    for i in range(dep.n_estimated):  # in this case dep.n_estimated = dep.n_detected
        assert np.allclose(
            dep.clusterers[i]._centers.ravel(), dep.detection_result.centers[i]
        )
        assert dep.clusterers[i]._centers.size == 3


def test_mem_cols_subset_of_det_cols_uses_centers_in_clustering():
    dep = DEP(
        detector=CountPeakDetector(
            bin_shape=[0.7, 0.7, 0.1],
            min_score=7,
        ),
        clusterer=SHDBSCAN(
            auto_allow_single_cluster=True,
            min_cluster_size=50,
            noise_proba_mode="conservative",
            cluster_proba_mode="hard",
            # scaler=None so it does not scale the detected centers
            # for clustering and they can be directly compared
            scaler=None,
        ),
        estimator=DBME(
            pdf_estimator=HKDE(
                bw=RuleOfThumbSelector(rule="scott"),
            ),
            kernel_calculation_mode="same",
        ),
        det_cols=["pmra", "pmdec", "parallax"],
        mem_cols=["pmra", "pmdec"],
    ).fit(sample2cdf)
    assert sorted(dep.mem_cols) == sorted(["pmra", "pmdec"])
    for i in range(dep.n_estimated):  # in this case dep.n_estimated = dep.n_detected
        assert np.allclose(
            dep.clusterers[i]._centers.ravel(), dep.detection_result.centers[i][:2]
        )
        assert dep.clusterers[i]._centers.size == 2


def test_no_err_or_corr_used():
    df = sample2cdf.copy()
    df = df[
        [
            col
            for col in df.columns
            if not col.endswith("_error") and not col.endswith("_corr")
        ]
    ]
    dep = DEP(
        detector=CountPeakDetector(
            bin_shape=[0.7, 0.7, 0.1],
            min_score=7,
        ),
        estimator=DBME(
            pdf_estimator=HKDE(
                bw=RuleOfThumbSelector(rule="scott"),
            ),
            kernel_calculation_mode="same",
        ),
        det_cols=["pmra", "pmdec", "parallax"],
        mem_cols=["pmra", "pmdec"],
    ).fit(df)
    # check if first covariance used is calculated only from bandwidth selection
    # because error and correlations were not included in the dataset
    used_data = dep.estimators[0]._estimators[0]._data
    used_covariance = dep.estimators[0]._estimators[0]._covariances[0]
    bw = RuleOfThumbSelector(rule="scott").get_bw(used_data)
    assert np.allclose(used_covariance, bw)


def test_cero_detected():
    dep = DEP(
        detector=CountPeakDetector(
            bin_shape=[0.7, 0.7, 0.1],
            # huge min_score so none
            # of the peaks pass the threshold
            min_score=1000,
        ),
        estimator=DBME(
            pdf_estimator=HKDE(
                bw=RuleOfThumbSelector(rule="scott"),
            ),
            kernel_calculation_mode="same",
        ),
        det_cols=["pmra", "pmdec", "parallax"],
        mem_cols=["pmra", "pmdec"],
    ).fit(sample2cdf)
    # all field probabilities
    assert dep.proba.shape == (sample2cdf.shape[0], 1)
    assert np.allclose(dep.proba, 1)


def test_cero_estimated():
    dep = DEP(
        detector=CountPeakDetector(
            bin_shape=[0.7, 0.7, 0.1],
            # huge min_score so none
            # of the peaks pass the threshold
            min_score=1000,
        ),
        estimator=DBME(
            pdf_estimator=HKDE(
                bw=RuleOfThumbSelector(rule="scott"),
            ),
            kernel_calculation_mode="same",
        ),
        tests=[
            # impassable test so no clusters are
            # estimated
            HopkinsTest(pvalue_threshold=0.00000001),
        ],
        test_cols=[["ra", "dec"]],
        det_cols=["pmra", "pmdec", "parallax"],
        mem_cols=["pmra", "pmdec"],
    ).fit(sample2cdf)
    # all field probabilities
    assert dep.proba.shape == (sample2cdf.shape[0], 1)
    assert np.allclose(dep.proba, 1)


def test_multiple_estimators_configuration():
    df = sample2cdf.copy()
    dep1 = DEP(
        # # Detector configuration for the detection step
        det_cols=["pmra", "pmdec", "parallax"],
        detector=CountPeakDetector(
            bin_shape=[0.7, 0.7, 0.1],
            min_score=6,
            max_n_peaks=1,
        ),
        sample_sigma_factor=1.5,
        mem_cols=["pmra", "pmdec"],
        estimator=DBME(
            pdf_estimator=HKDE(
                bw=RuleOfThumbSelector(rule="scott", diag=True),
            ),
            kernel_calculation_mode="per_class",
        ),
    ).fit(df)

    fe1 = dep1.estimators[0]._estimators[0]
    fbw1 = fe1._base_bw
    ce1 = dep1.estimators[0]._estimators[1]
    cbw1 = ce1._base_bw

    bw1 = [fbw1[0, 0], fbw1[1, 1]]
    bw2 = [cbw1[0, 0], cbw1[1, 1]]

    dep2 = DEP(
        # # Detector configuration for the detection step
        det_cols=["pmra", "pmdec", "parallax"],
        detector=CountPeakDetector(
            bin_shape=[0.7, 0.7, 0.1],
            min_score=6,
            max_n_peaks=1,
        ),
        sample_sigma_factor=1.5,
        mem_cols=["pmra", "pmdec"],
        estimator=DBME(
            pdf_estimator=[
                HKDE(bw=bw1),
                HKDE(bw=bw2),
            ],
            kernel_calculation_mode="per_class",
        ),
    ).fit(df)

    fe1 = dep1.estimators[0]._estimators[0]
    fcov1 = fe1._covariances[0]
    ce1 = dep1.estimators[0]._estimators[1]
    ccov1 = ce1._covariances[0]

    fe2 = dep2.estimators[0]._estimators[0]
    fcov2 = fe2._covariances[0]
    ce2 = dep2.estimators[0]._estimators[1]
    ccov2 = ce2._covariances[0]

    assert np.allclose(fcov1, fcov2)
    assert np.allclose(ccov1, ccov2)

    p1 = dep1.proba_df()
    p2 = dep2.proba_df()
    assert np.allclose(p1, p2)
