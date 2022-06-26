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
def three_clusters_sample():
    field_size = int(1e4)
    cluster_size = int(1e2)
    field = StarField(
        pm=multivariate_normal(mean=(0.0, 0.0), cov=20),
        space=UniformFrustum(locs=(120.5, -27.5, 12), scales=(1, 1, -11.8)),
        n_stars=field_size,
    )
    clusters = [
        StarCluster(
            space=multivariate_normal(
                mean=polar_to_cartesian([120.7, -28.5, 1.15]), cov=0.5
            ),
            pm=multivariate_normal(mean=(0.5, 0), cov=1.0 / 10),
            n_stars=cluster_size,
        ),
        StarCluster(
            space=multivariate_normal(
                mean=polar_to_cartesian([120.8, -28.6, 5]), cov=0.5
            ),
            pm=multivariate_normal(mean=(4.5, 4), cov=1.0 / 10),
            n_stars=cluster_size,
        ),
        StarCluster(
            space=multivariate_normal(
                mean=polar_to_cartesian([120.9, -28.7, 8]), cov=0.5
            ),
            pm=multivariate_normal(mean=(7.5, 7), cov=1.0 / 10),
            n_stars=cluster_size,
        ),
    ]
    df = Synthetic(star_field=field, clusters=clusters).rvs()
    return df


data = three_clusters_sample()[["pmra", "pmdec", "log10_parallax"]].values

# Detect density peaks in the data
result = CountPeakDetector(bin_shape=[0.5, 0.5, 0.05]).detect(data)
print(result)
# DetectionResult(
#     centers=numpy.ndarray([
#         [7.533341815598953, 7.002588300090162, 0.9019444495539226],
#         [4.566029697513946, 3.7463844905908914, 0.6991673208304323],
#         [0.3656786941060229, -0.0843682251106874, 0.060738437970968506]
#     ]),
#     sigmas=numpy.ndarray([
#         [0.5, 0.5, 0.05],
#         [0.5, 0.5, 0.05],
#         [0.5, 0.5, 0.05]
#     ]),
#     scores=numpy.ndarray([
#         7.019414816995393,
#         4.966786793237695,
#         3.908990867747345
#     ]),
#     counts=numpy.ndarray([24.6875, 28.90625, 20.65625]),
#     edges=numpy.ndarray([
#         [
#             [6.627436847329907, 7.627436847329907],
#             [6.261616766665554, 7.261616766665554],
#             [0.8047405780636238, 0.9047405780636238]
#         ],
#         [
#             [3.877436847329907, 4.877436847329907],
#             [3.0116167666655542, 4.011616766665554],
#             [0.6297405780636238, 0.7297405780636239]
#         ],
#         [
#             [-0.372563152670093, 0.627436847329907],
#             [-0.7383832333344458, 0.2616167666655542],
#             [-0.020259421936376268, 0.07974057806362374]
#         ]
#     ])
# )
# Note: the above print is actually made with prettyprinter.cpprint
# (https://pypi.org/project/prettyprinter/)
# and prettyprinter.install_extras(),
# it is used only for demonstration purposes.
