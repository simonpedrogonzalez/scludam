import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal.windows import gaussian as gaus_win

from scludam import extend_1dmask

win = gaus_win(5, 1)
print(win)
# [0.13533528 0.60653066 1.         0.60653066 0.13533528]
mask = extend_1dmask(win, 2)
sns.heatmap(mask, annot=True)
plt.show()
