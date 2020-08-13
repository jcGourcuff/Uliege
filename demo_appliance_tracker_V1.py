import numpy as np
import pandas as pd
from utils.display import Graphs
from dataManager.load import DataSets
from nilm.filters import Filters
from utils.compute import Metrics
from nilm.appliance_tracker_V1 import Tracker

Window_size = 1440

# loads demo data. 5 appliances, timestamp in secondes sampled every ~3 secondes.
df = DataSets.REDD(1, demo=True)
df.head()


# resample the data to have minute index
sampled = df.resample('1min').mean().fillna(method='ffill')
sampled.head()

# select 3 appliances
test = sampled[['refrigerator', 'lighting', 'dishwasher']]

# select 14 days of data
days = [test.iloc[Window_size * k:Window_size * (k + 1)] for k in range(14)]

# visualize
Graphs.aggregate(days[0], title='Aggregated consumption - day 1')
Graphs.decompose_aggregate(days[0], title='Disaggregated consumption - day 1')

# compute aggregated power
aggregated = [d.sum(axis=1) for d in days]

# Choose parameters for the filters. The parameters have been chosen using Filter_tuning.grid_search
params = {'nb_iter': 5, 'sigma_r': 60, 'flat_tresh': 10, 'sharp_tresh': 60,
          'min_sharp_tresh': 10, 'sharp_rate': .75, 'med_k_size': 3, 'bil_k_size': 3, 'sigma_s': 5}


# filter the signals. See nilm.filters.py file for more info.
filtered = [Filters.apply_filters(agg, **params)[2] for agg in aggregated]

# visualize
Graphs.draw(filtered[0], title='Filtered signal - day 1')

# Instanciate a tracker and initiate a list to hold the results
T = Tracker(T_error=.2, T_detect=60, T_proba=.9)
results = []

# loop over the days - The distributions are updated after every day
for k, x in enumerate(filtered):
    results.append(T.track(x))
    n_app = T.n_app
    T.update_distribs()
    Graphs.decompose_aggregate(
        days[k][days[k].columns[::-1]], title='Ground truth - day ' + str(k + 1))
    Graphs.decompose_aggregate(
        results[-1], title='Infered - day ' + str(k + 1))
    print("Number of appliances went from {} to {} after {}th day".format(
        n_app, T.n_app, k + 1))
