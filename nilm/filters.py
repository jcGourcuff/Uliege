import numpy as np
import pandas as pd
from scipy.signal import medfilt
from utils.display import Graphs
import matplotlib.pyplot as plt
from utils.compute import Functions


class Filters():
    """
    Implements an adapted version of the filtering framework proposed in
    Rodriguez-Silva, A., & Makonin, S. (2019). Universal Non-Intrusive
    Load Monitoring (UNILM) Using Filter Pipelines, Probabilistic Knapsack,
    and Labelled Partition Maps. Asia-Pacific Power and Energy Engineering
    Conference, APPEEC, 2019-Decem. https://doi.org/10.1109/APPEEC45492.2019.8994618
    """

    def __init__(self):
        """
        All methods in this class are static.
        """
        return None

    @staticmethod
    def apply_df_filter(df, sigma_r=60, nb_iter=2, flat_tresh=10, sharp_tresh=60, min_sharp_tresh=10, sharp_rate=.75,
                        med_k_size=3, bil_k_size=3, sigma_s=5):
        """
        Applies Filters.apply_filters() on all columns.

        :param df: Pandas DataFrame. The data to filter.
        :return: Pandas DataFrame. The filtered signals.

        ..seealso: Filters.apply_filters().
        """
        filtered = []
        for name in df.columns:
            _, _, result = Filters.apply_filters(
                df[name], sigma_r, nb_iter, flat_tresh, sharp_tresh, min_sharp_tresh, sharp_rate, med_k_size,
                bil_k_size, sigma_s)
            filtered.append(result)
        new_df = pd.concat(filtered, axis=1)
        new_df.index = df.index
        new_df.columns = df.columns
        return new_df

    @staticmethod
    def bilat_sharp(input, bil_k_size=3, sigma_s=10, sigma_r=60, sharpen=60):
        """
        Applies bilateral + sharpening filtering.

        :param input: Pandas Series.
        :return: Pandas Series. Filtered by bilateral and sharpenning filters with specified parmaeters.

        ..seealso: Filters.bilateral_filter(), Filters.edge_sharpening()
        """

        index = input.index
        x = input.to_numpy()
        x = Filters.bilateral_filter(x, index, sigma_s, sigma_r, bil_k_size)
        x = Filters.edge_sharpening(x, treshold=sharpen)
        x = pd.Series(x)
        x.index = index
        return x

    @staticmethod
    def apply_filters(input, nb_iter=2, sigma_r=60, flat_tresh=10, sharp_tresh=60, min_sharp_tresh=10, sharp_rate=.75,
                      med_k_size=3, bil_k_size=3, sigma_s=5, show=False):
        """
        Applies the complete filter framework to the input. The signal goes through
        a median filter and bilateral filtering and sharpening for nb_iter times.

        :param input: Pandas Series. The signal to filter.
        :param nb_iter: Integer. The number of time Filters.bilat_sharp() is applied to the signal.
        :param min_sharp_tresh: Minimum value for the sharpenning treshold.
        :param sharpe_rate: Float. Sharpening treshold decrease rate.
        :param show: Boolean. If True, evolution of signal through filtering is shown after the function call.
        :return: (Pandas Series, list, Pandas Series). The residuals, the mean
                 squarred errors between to consecutive filtering step and the filtered signal.

        ..seealso: Filters.bilat_sharp(), Filters.median_filter(), Filters.edge_sharpening(),
                   Filters.bilateral_filter(), Filters.flatten().
        """

        filt_loss = []
        x = pd.Series(Filters.median_filter(
            input.to_numpy(), kernel_size=med_k_size))
        x.index = input.index
        last = input
        level = sharp_tresh
        if show:
            plt.figure(figsize=(20, 8))
        for k in range(nb_iter):
            if k % 2 == 0:
                flattened = pd.Series(Filters.flatten(
                    x.to_numpy(), treshold=flat_tresh))
                flattened.index = x.index
                x = flattened
                if show:
                    plt.plot(x, color='b', alpha=.25 + k * .75 / nb_iter)
            x = Filters.bilat_sharp(
                x, bil_k_size=bil_k_size, sigma_s=sigma_s, sigma_r=sigma_r, sharpen=level)
            error = x - last
            filt_loss.append(error.std())
            last = x

            if level > min_sharp_tresh:
                level = int(level * sharp_rate)

        flattened = pd.Series(Filters.flatten(
            last.to_numpy(), treshold=flat_tresh))
        flattened.index = last.index
        last = flattened

        energy_loss = input - last
        if show:
            plt.show()
        return energy_loss, filt_loss, last

    @staticmethod
    def flatten(input, treshold):
        """
        Flattens the trabsitions that are below the given treshold.

        :param input: Numpy array. Signal to flatten.
        :param treshold: Value under wich transitions are flattened.
        :return: Numpy array. The flattened signal.
        """
        result = input
        for k in range(len(result) - 1):
            if np.abs(result[k + 1] - result[k]) < treshold:
                result[k + 1] = result[k]

        return result

    @staticmethod
    def median_filter(input, kernel_size):
        """
        Apply the classic median filter to remove the spikes.

        :param input: Numpy array. Data to filter.
        :param kernel_size: Integer. Size of the filter kernel.
        :return: Numpy arry. Filtered signal.
        """
        return medfilt(input, kernel_size=kernel_size)

    @staticmethod
    def bilateral_filter(input, timestamps, sigma_s, sigma_r, kernel_size):
        """
        Implementation of 1-d bilateral filtering. Window_size is index wisea
        and not time wise but distance between data point is computed time wise.

        :param: input: Numpy array. The data to filter.
        :param timestamps: Numpy array. The timestamps of the data.
        :param sigma_s: Caracteristic window range. (Note : Caracteristic window range is in fact sigma_s*kernel_size)
        :param sigma_r: Caracteristic power difference under which the signal is smoothed.
        :param kernel_size: Integer. Size of the base kernel.
        :return: Numpy array. The filtered signal.

        """
        window_size = int(kernel_size * sigma_s)
        Gs = Functions.gaussian(sig=sigma_s)
        Gr = Functions.gaussian(sig=sigma_r)

        filtered = []
        for k in range(len(input)):
            factor = 0
            sum = 0
            for l in range(max(0, k - window_size), min(len(input), k + window_size)):
                a = Gs(np.abs(timestamps[k] - timestamps[l]).seconds / 60
                       ) * Gr(np.abs(input[k] - input[l]))
                factor += a
                sum += a * input[l]
            filtered.append(sum / factor)
        return np.array(filtered)

    @staticmethod
    def edge_sharpening(input, treshold):
        """
        Merges consecutives edges if the first one exeeds the treshold.

        :param input: Numpy array. Signal to filter.
        :param treshold: Treshold over which an edge is considered as significant.
        :return: Numpy array. Shrapened signal.
        """
        signal = input
        filtered = [signal[0]]
        lagged_plus = signal[1:]
        lagged_minus = signal[:-1]
        dy = lagged_plus - lagged_minus

        i = 0
        while (i < len(dy) - 1):
            value = dy[i]
            if np.abs(value) > treshold:
                if dy[i] * dy[i + 1] <= 0:
                    filtered.append(lagged_plus[i])
                    i += 1
                else:
                    mem = i
                    while i < len(dy) - 1 and dy[mem] * dy[i] > 0:
                        i += 1

                    filtered = filtered + ([lagged_minus[i]] * (i - mem))

            else:
                if np.abs(dy[i + 1]) > treshold:
                    i += 1
                    filtered.append(lagged_minus[i])
                else:
                    mem = i
                    while i < len(dy) - 1 and np.abs(dy[i]) < treshold:
                        i += 1

                    filtered = filtered[:-1] + list(Filters.median_filter(
                        lagged_minus[mem:i + 1], kernel_size=3))

        filtered.append(filtered[-1])
        return np.array(filtered)
