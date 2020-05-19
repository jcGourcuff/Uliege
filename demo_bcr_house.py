"""
BCR, 2020-05-19
Adapted from demo.py from Jean to run on my house's data. My data is not labeled.
"""

from dataManager.load import DataSets
from nilm.appliance_tracker import Tracker
from nilm.filters import Filters
from utils.compute import Metrics
from utils.display import Graphs

# load BCR's house data
df = DataSets.BCR(1)
df.head()

# resample the data to have minute index
sampled = df.resample('1min').mean()
sampled.head()

# select a range of data between two dates (typically one day span)
test = sampled.loc['2020-5-18':'2020-5-19']

# visualize
Graphs.aggregate(test)
# Graphs.decompose_aggregate(test)

# compute aggregated power
aggregated = test.sum(axis=1)

# filter the signal. See nilm.filters.py file for more info.
energy_loss, filt_loss, filtered = Filters.apply_filters(aggregated, sigma_r=60, sigma_s=5, nb_iter=5, flat_tresh=10,
                                                         sharp_tresh=60, min_sharp_tresh=10, sharp_rate=.75, show=False)

# visualize
Graphs.draw(filtered, title='Filtered signal')
Graphs.draw(energy_loss, title='Residuals')

# Instanciate a tracker and disaggregate the signal
T = Tracker(T_detect=60, T_error=.1)
result = T.track(filtered)
Graphs.decompose_aggregate(result)

# Some metrics.

Energy = Metrics.E(aggregated, msg=True)

Energy_apps = Metrics.E_apps(test)
Energy_apps

residual, mean, std, percent = Metrics.E_filt(aggregated, filtered, msg=True)
Graphs.draw(residual)
