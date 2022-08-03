import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from scludam import DBME, SHDBSCAN
from scludam.synthetic import (
    BivariateUniform,
    StarCluster,
    StarField,
    Synthetic,
    UniformFrustum,
    polar_to_cartesian,
)

# generate some data

fmix = 0.9
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
df = Synthetic(star_field=field, clusters=clusters).rvs()

data = df[["pmra", "pmdec"]].values

# create some random observational error
random_error = np.random.normal(0, 0.1, data.shape)

# calculate some initial probabilities
shdbscan = SHDBSCAN(
    min_cluster_size=150, noise_proba_mode="outlier", auto_allow_single_cluster=True
).fit(data)

# use DBME to fit HKDE models and calculate membership probabilities
dbme = DBME().fit(data, shdbscan.proba, random_error)
print(dbme.posteriors)
# [[9.45802647e-01 5.41973532e-02]
#  ...
#  [2.77988823e-01 7.22011177e-01]]

# plot to compare initial probabilities with membership probabilities
shdbscan.surfplot(cols=["pmra", "pmdec"])
dbme.surfplot(cols=["pmra", "pmdec"])
plt.show()
