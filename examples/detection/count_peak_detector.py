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
print(result)
# DetectionResult(
#     centers=numpy.ndarray([
#         [4.536048743232476, 3.9186486456763276, 0.6994997601345216],
#         [7.538194372795517, 6.998371367057486, 0.9020709230859785],
#         [0.4479574532248377, -0.01917758842077178, -0.3009924913261566]
#     ]),
#     sigmas=numpy.ndarray([
#         [0.5, 0.5, 0.05],
#         [0.5, 0.5, 0.05],
#         [0.5, 0.5, 0.05]
#     ]),
#     scores=numpy.ndarray([
#         6.9706701775336,
#         6.627355856978734,
#         5.886588680967625
#     ]),
#     counts=numpy.ndarray([35.21875, 24.71875, 31.21875]),
#     edges=numpy.ndarray([
#         [
#             [4.016260095931324, 5.016260095931324],
#             [3.1871460374559666, 4.187146037455967],
#             [0.6297442445624907, 0.7297442445624908]
#         ],
#         [
#             [6.766260095931324, 7.766260095931324],
#             [6.437146037455967, 7.437146037455967],
#             [0.8047442445624906, 0.9047442445624907]
#         ],
#         [
#             [-0.23373990406867584, 0.7662600959313242],
#             [-0.5628539625440334, 0.4371460374559666],
#             [-0.37025575543750955, -0.27025575543750957]
#         ]
#     ]),
#     offsets=numpy.ndarray([
#         [0.0, 0.25, 0.0],
#         [0.25, 0.0, 0.025],
#         [0.25, 0.0, 0.0]
#     ]),
#     indices=numpy.ndarray([[27, 21, 28], [32, 28, 31], [18, 14, 8]])
# )
# Note: the above print is actually made with (https://pypi.org/project/prettyprinter/)

# Plot the third peak found in pmra vs pmdec
detector.plot(2)
plt.show()
