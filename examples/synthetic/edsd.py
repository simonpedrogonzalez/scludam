import matplotlib.pyplot as plt
import seaborn as sns

from scludam.synthetic import EDSD

edsd = EDSD(w0=0, wl=5, wf=14)
data = edsd.rvs(1000)
sns.distplot(data, bins=100)
plt.show()
