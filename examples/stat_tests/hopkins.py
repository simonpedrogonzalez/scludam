from sklearn.datasets import load_iris

from scludam import HopkinsTest

iris = load_iris().data

ht = HopkinsTest().test(data=iris)
print(ht)
#  >> HopkinsTestResult(rejectH0=True, value=0.9741561806256851, pvalue=2.220446049250313e-16)
