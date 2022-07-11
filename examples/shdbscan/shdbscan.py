import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from scludam import SHDBSCAN

iris = load_iris()

shdbscan = SHDBSCAN(min_cluster_size=20).fit(iris.data)

print(shdbscan.proba.round(2))
# [[0.   1.   0.  ]
#  [0.02 0.95 0.03]
#  ...
#  [0.18 0.08 0.74]
#  [0.03 0.05 0.92]]
print(shdbscan.labels)
# [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1,  0,
#   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
# ...
#   1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1, -1,
#   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]

shdbscan.pairplot(
    diag_kind="hist",
    palette="copper",
    corner=True,
    cols=iris.feature_names,
    diag_kws={"bins": 20},
)
plt.show()
