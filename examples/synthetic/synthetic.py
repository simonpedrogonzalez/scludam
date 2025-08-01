import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

from scludam.synthetic import (
    StarCluster,
    StarField,
    Synthetic,
    UniformFrustum,
    polar_to_cartesian,
)

data = Synthetic(
    star_field=StarField(
        pm=multivariate_normal((-5, 6), cov=3.0),
        space=UniformFrustum(locs=(118, -31, 8), scales=(2, 2, -7)),
        n_stars=1000,
    ),
    clusters=[
        StarCluster(
            space=multivariate_normal(
                mean=polar_to_cartesian([118.75, -29.75, 1.5]), cov=1 / 3
            ),
            pm=multivariate_normal(mean=(-5.4, 6.75), cov=1.0 / 34),
            n_stars=50,
        ),
        StarCluster(
            space=multivariate_normal(
                mean=polar_to_cartesian([119.25, -30.5, 2]), cov=1 / 3
            ),
            pm=multivariate_normal(mean=(-6.25, 7.75), cov=1.0 / 34),
            n_stars=50,
        ),
    ],
    representation_type="spherical",
).rvs()

fig, ax = plt.subplots(ncols=2)
sns.scatterplot(
    data=data, x="pmra", y="pmdec", hue="p_pm_field", palette="viridis", ax=ax[0]
)
sns.scatterplot(
    data=data, x="ra", y="dec", hue="p_space_field", palette="viridis", ax=ax[1]
)
plt.show()
