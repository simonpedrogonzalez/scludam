import matplotlib.pyplot as plt

from scludam import DEP, CountPeakDetector, HopkinsTest, Query, RipleysKTest

# search some data from GAIA
# with error and correlation
# columns for better KDE
df = (
    Query()
    .select(
        "ra",
        "dec",
        "ra_error",
        "dec_error",
        "ra_dec_corr",
        "pmra",
        "pmra_error",
        "ra_pmra_corr",
        "dec_pmra_corr",
        "pmdec",
        "pmdec_error",
        "ra_pmdec_corr",
        "dec_pmdec_corr",
        "pmra_pmdec_corr",
        "parallax",
        "parallax_error",
        "parallax_pmra_corr",
        "parallax_pmdec_corr",
        "ra_parallax_corr",
        "dec_parallax_corr",
        "phot_g_mean_mag",
    )
    # search for this identifier in simbad
    # and bring data in a circle of radius
    # 1/2 degree
    .where_in_circle("ngc2168", 0.5)
    .where(("parallax", ">", 0))
    .where(("phot_g_mean_mag", "<", 18))
    # include some common criteria
    # for data precision
    .where_arenou_criterion()
    .where_aen_criterion()
    .get()
    .to_pandas()
)

# Build Detection-Estimation Pipeline
dep = DEP(
    # Detector configuration for the detection step
    detector=CountPeakDetector(
        bin_shape=[0.3, 0.3, 0.07],
        min_score=3,
    ),
    det_cols=["pmra", "pmdec", "parallax"],
    sample_sigma_factor=2,
    # Clusterability test configuration
    tests=[
        RipleysKTest(pvalue_threshold=0.05, max_samples=100),
        HopkinsTest(),
    ],
    test_cols=[["ra", "dec"]] * 2,
    # Membership columns to use
    mem_cols=["pmra", "pmdec", "parallax", "ra", "dec"],
).fit(df)

# plot the results
dep.scatterplot(["pmra", "pmdec"])
plt.show()

# write results to file
dep.write("ngc2168_result.fits")
