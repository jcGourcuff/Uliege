import numpy as np
import pandas as pd
from utils.display import Graphs
from dataManager.load import DataSets
from nilm.filters import Filters
from utils.compute import Metrics
from nilm.appliance_tracker_V2 import Tracker

window_size = 1440

#loads demo data. 5 appliances, timestamp in secondes sampled every ~3 secondes.
df = DataSets.REDD(1, demo = True)
df.head()


#resample the data to have minute index  and 3 appliances
sampled = df.resample('1min').mean().fillna(method = 'ffill')[['refrigerator', 'lighting', 'dishwasher']]
sampled.head()

#select two weeks of data
days = [sampled.iloc[window_size * k:window_size *(k + 1)] for k in range(14)]

#visualize first day
Graphs.aggregate(days[0])
Graphs.decompose_aggregate(days[0])

#compute aggregated signals
aggregated = [d.sum(axis=1) for d in days]

#Set the filtering parameters and proced to filter the signals
params = {'nb_iter': 5, 'sigma_r': 60, 'flat_tresh': 10, 'sharp_tresh': 60,
          'min_sharp_tresh': 10, 'sharp_rate': .75, 'med_k_size': 3, 'bil_k_size': 3, 'sigma_s': 5}
filtered = [Filters.apply_filters(agg, **params)[2] for agg in aggregated]

#visualize
Graphs.draw(filtered[0], title = 'Filtered signal')

#Instanciate a tracker and a list to hold the results
T = Tracker(aggregated = filtered[0], T_detect=60, T_error=.1, train_window = 15, gamma_rkp = 2)
results = []
results.append(T.track(train_rate = 1, batch_size = 10, n_batch = 10))

for f in filtered[1:7] :
    T.add_new_data(f)
    results.append(T.track(train_rate =5, batch_size = 10, n_batch = 10))


#Let's visualize the results
for k in range(7):
    Graphs.draw(filtered[k])
    Graphs.decompose_aggregate(days[k])
    Graphs.decompose_aggregate(results[k])

#Let's see how the models learned
apps_metrics = [Metrics.evaluate(a.gt_log, np.array(a.prediction_log) > .5) for a in T.dictionary]
for app in apps_metrics :
    print(app)
