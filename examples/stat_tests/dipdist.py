import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

from scludam import DipDistTest

iris = load_iris().data

dd = DipDistTest().test(data=iris)
print(dd)
#  >> DipDistTestResult(rejectH0=True, value=0.01415416887708115, pvalue=0.0, dist=array([0.1, ..., 7.08519583]))
sns.histplot(dd.dist, bins=100)
plt.show()
