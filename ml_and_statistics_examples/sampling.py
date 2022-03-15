import numpy as np
import pandas as pd
import plotly.express as px
from dstk import dviztk

# log scale sampling ###################################################################################################

x = 10 ** (-3 * np.random.rand(10_000))
px.histogram(x).plot()

print(10 ** -0, 10 ** -1, 10 ** -2, 10 ** -3)
print((x < 0.1).mean(), (x < 0.01).mean(), (x < 1).mean())
min(x), max(x)
