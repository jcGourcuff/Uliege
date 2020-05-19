
import numpy as np
import pandas as pd
from utils.display import Graphs
import matplotlib.pyplot as plt
from dataManager.load import DataSets
from nilm.filters import Filters
from utils.compute import Metrics
from utils.compute import Functions

df = DataSets.REDD(1)

appliances = list(df.columns)
sampled = Functions.resample(df, rate =60)

window_size = 60
nb_iter = 1000
nb_app = 10
index_max = sampled.shape[0]

M = Metrics()

means = []
stds = []
percents = []
residuals = pd.Series([0]*window_size)

bad_cases = []

for k in range(nb_iter):
    inf = np.random.randint(0,index_max-window_size)
    sup = inf+window_size-1
    random_apps = np.random.choice(appliances, nb_app)

    test = sampled.loc[inf:sup,random_apps]

    aggregate = test.sum(axis=1)
    energy_loss, filt_loss, filtered = Filters.apply_filters(aggregate,  sigma_r = 60, sigma_s = 5, nb_iter = 5, flat_tresh = 10, sharp_tresh = 60, min_sharp_tresh = 10, sharp_rate = .75)

    residual, mean, std, percent = M.E_filt(aggregate, filtered)

    if std > 100 :
        bad_cases.append((inf,sup,random_apps))

    means.append(mean)
    stds.append(std)
    percents.append(percent)
    residual.reset_index(inplace = True, drop=True)
    residuals = residuals + residual

residuals = residuals/nb_iter


residuals
plt.plot(np.arange(nb_iter), means)
plt.plot(np.arange(nb_iter), stds)
plt.plot(np.arange(nb_iter), percents)
Graphs.draw(residuals)
np.std(stds)

bad_results = pd.DataFrame(bad_cases)
bad_results
