import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from scludam import SHDBSCAN

iris = load_iris()

shdbscan = SHDBSCAN(min_cluster_size=20, outlier_quantile=0.8).fit(iris.data)

shdbscan.outlierplot(bins=20, color='k')
plt.show()
