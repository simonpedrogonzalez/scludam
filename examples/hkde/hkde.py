import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from scludam import HKDE

iris = load_iris()

hkde = HKDE().fit(iris.data)

print(hkde.pdf(iris.data))
# [1.41917213e+00 4.68331703e-01 4.92541896e-01 1.03268828e+00
# ...
#  4.35570423e-01 1.72914477e-01]
hkde.plot()
plt.show()
