import numpy as np
import pandas as pd
from utils.display import Graphs
import matplotlib.pyplot as plt
from dataManager.load import DataSets
from nilm.filters import Filters
from utils.compute import Metrics
from utils.compute import Functions
from nilm.appliance_tracker import Tracker

#loads demo data. 5 appliances, timestamp in secondes sampled every ~3 secondes.
df = DataSets.REDD(1, demo = True)
appliances = list(df.columns)
df.head()

#resample the data to have minute index
sampled = Functions.resample(df, rate =60)
sampled.head()

#select a day of data and 3 appliances
test = sampled[:24*60][['refrigerator', 'lighting', 'dishwasher']]

#visualize
Graphs.aggregate(test)
Graphs.decompose_aggregate(test)

#compute aggregated power
aggregated = test.sum(axis = 1)

#filter the signal. See nilm.filters.py file for more info.
energy_loss, filt_loss, filtered = Filters.apply_filters(aggregated,  sigma_r = 60, sigma_s = 5, nb_iter = 5, flat_tresh = 10, sharp_tresh = 60, min_sharp_tresh = 10, sharp_rate = .75, show = False)

#visualize
Graphs.draw(filtered, title = 'Filtered signal')
Graphs.draw(energy_loss, title = 'Residuals')

#or filter the data set column by column. Since the operation is costly, we only do it for the first 6 hours.
individual_filt = Filters.apply_df_filter(test[:6*60],  sigma_r = 60, sigma_s = 10, nb_iter = 5, flat_tresh = 10, sharp_tresh = 60, min_sharp_tresh = 10, sharp_rate = .75)
Graphs.decompose_aggregate(individual_filt)

#Instanciate a tracker and disaggregate the signal
T = Tracker(T_detect = 60, T_error = .1)
result = T.track(filtered)
Graphs.decompose_aggregate(result)

#Some metrics.

Energy = Metrics.E(aggregated, msg = True)

Energy_apps = Metrics.E_apps(test)
Energy_apps

residual, mean, std, percent = Metrics.E_filt(aggregated, filtered, msg = True)
Graphs.draw(residual)

residuals, percents = Metrics.E_filt_apps(test[:6*60], individual_filt)
