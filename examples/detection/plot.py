import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from scludam import CountPeakDetector
from scludam.synthetic import (
    StarCluster,
    StarField,
    Synthetic,
    UniformFrustum,
    polar_to_cartesian,
)

# Generate some data
np.random.seed(134)

field_size = int(1e4)
cluster_size = int(1e2)
field = StarField(
    pm=multivariate_normal(mean=(0.0, 0.0), cov=(20, 15)),
    space=UniformFrustum(locs=(120.5, -27.5, 12), scales=(1, 1, -11.8)),
    n_stars=field_size,
)
clusters = [
    StarCluster(
        space=multivariate_normal(
            mean=polar_to_cartesian([120.7, -28.5, 0.5]), cov=0.5
        ),
        pm=multivariate_normal(mean=(0.5, 0), cov=1.0 / 10),
        n_stars=cluster_size,
    ),
    StarCluster(
        space=multivariate_normal(mean=polar_to_cartesian([120.8, -28.6, 5]), cov=0.5),
        pm=multivariate_normal(mean=(4.5, 4), cov=1.0 / 10),
        n_stars=cluster_size,
    ),
    StarCluster(
        space=multivariate_normal(mean=polar_to_cartesian([120.9, -28.7, 8]), cov=0.5),
        pm=multivariate_normal(mean=(7.5, 7), cov=1.0 / 10),
        n_stars=cluster_size,
    ),
]
df = Synthetic(star_field=field, clusters=clusters).rvs()

# Select the data to be used for the detection
data = df[["pmra", "pmdec", "log10_parallax"]].values

# Detect density peaks in the data
detector = CountPeakDetector(bin_shape=[0.5, 0.5, 0.05])
result = detector.detect(data)

# Custom plot of third peak scores in pmdec and log10_parallax
detector.plot(
    peak=2,
    x=1,
    y=2,
    mode="s",
    cols=["pmra", "pmdec", "log10_parallax"],
    annot_threshold=5,
    cmap="copper",
)
plt.show()
# Custom plot of third peak background in pmra and pmdec
detector.plot(
    peak=2,
    x=0,
    y=1,
    mode="b",
    cols=["pmra", "pmdec", "log10_parallax"],
    annot=False,
    cmap="copper",
)
plt.show()
