import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scludam.synthetic import UniformFrustum, cartesian_to_polar

# parallax between 14 and 4 arcsec
uf = UniformFrustum(locs=(120, -80, 14), scales=(1, 1, -10))
# points in cartesian coordinates in parsecs
data = uf.rvs(1000)
# distribution in spherical coordinates
sns.pairplot(pd.DataFrame(cartesian_to_polar(data), columns=["ra", "dec", "plx"]))
plt.show()
