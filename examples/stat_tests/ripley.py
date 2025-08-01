import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

from scludam import RipleysKTest

iris = load_iris().data

rk = RipleysKTest().test(data=iris[:, 0:2])
print(rk)
#  >> RipleyKTestResult(rejectH0=True, value=0.07193140227503197, radii=array([0., ..., 0.25]), l_function=array([0., ..., 0.31045381]))
sns.lineplot(
    x=rk.radii,
    y=rk.l_function,
    linestyle="--",
    color="red",
    label="Estimated L function",
)
sns.lineplot(
    x=rk.radii,
    y=rk.radii,
    color="k",
    label="Theoretical Poisson Point Process L function",
)
plt.show()
