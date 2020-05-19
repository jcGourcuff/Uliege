import numpy as np
import pandas as pd


class Functions:
    """
    This class implements a few useful functions.
    """

    def __init__(self):
        """
        All methods in this class are static.
        """
        return None

    @staticmethod
    def gaussian(sig=1., mu=0., normalized=False):
        """
        Return a gaussian function with the specified parameters.

        :param sig: Gaussian's standard deviation. Default 1.
        :param mu: Gaussian's mean. Default 0.
        :param normalized: If True, normailzed for L1 norm.
        :return: A callable function.
        """
        factor = 1.
        if normalized:
            factor = 1 / np.sqrt(2 * np.pi)

        def f(x):
            return (1. / factor) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        return f

    @staticmethod
    def resample(df, rate):
        """
        Method to resample data sets that have second timestamps. Resample to the given rate.
        e.g. rate = 60 will resample the dataset to 1 min intervalls between each data sample.
        :param df: The data set to resample.
        :param rate: Integer. The number of seconds that must seperate each sample.
        :return: Pandas DataFrame.
        """

        nb_iter = 0
        new_index = []

        new_df = []

        mem = 0
        i = 1
        while i < df.shape[0] - 1:
            nb_iter += 1
            count = 0
            while count < rate and i < df.shape[0] - 1:
                count += df.index[i] - df.index[i - 1]
                i += 1

            new_df.append(df.iloc[mem:i].mean())
            mem = i
            new_index.append(nb_iter)

        new_df = pd.concat(new_df, axis=1, ignore_index=True).transpose()
        new_df.index = new_index
        return new_df


class Metrics():
    """
    Implements a set of metrics.
    """

    def __init__(unit='watt', time='min'):
        """
        Class containing only static methods.
        """
        return None

    @staticmethod
    def E(serie, msg=False):
        """
        Integrates the series with respect to the time. Thus returns the total energy
        consumed in the time window of the series.

        :param serie: Pandas Series. The power to integrate.
        :param msg: If True, calling the function will display the result.
        :return: Integer. The result rounded to the unit.
        """

        x = serie.values
        total_power = np.sum(x) * 60 / 3600
        if msg:
            print(str(total_power) + " Wh")

        return int(total_power)

    @staticmethod
    def E_apps(df):
        """
        Calls Metrics.E() for each column of the data frame.

        :param df: The Pandas Dataframe containing the power series ton integrate.
        :return: A Pandas Series with the integrate value for each column.
        """
        result = pd.Series([Metrics.E(df[name]) for name in df.columns])
        result.index = df.columns
        return result

    @staticmethod
    def E_filt(input, output, msg=False):
        """
        Returns residuals = input-output, the mean and the standard deviation of the same residuals.
        Also returns the energy percentage of the input signal.

        :param input: Pandas Series. Signal before filtering.
        :param output: Pandas Series. Signal after filtering.
        :param msg: boolean. If True, calling this method will display the key results.
        :return: Tuple of the form (Pandas Series, float, float, float) respectively
                 corresponding to the residuals, their mean and std, and the
                 percentage of energy of the input that has been preserved through the filtering.
        """
        e_i = Metrics.E(input)
        e_o = Metrics.E(output)
        percent = 100
        if e_i > 0:
            percent = 100 * e_o / e_i
        residual = input - output
        mean, std = residual.mean(), residual.std()
        if msg:
            print("Filtered signal is {}% of the original".format(percent))
            print("Residuals : m = {} ; std = {}".format(mean, std))
        return residual, mean, std, percent

    @staticmethod
    def E_filt_apps(input, output):
        """
        Applies Metrics_filt.E_filt() to each columns of the dfs.

        :param input: Pandas DataFrame. Signals before filtering.
        :param output: Pandas DataFRame. Siganals after filtering.
        :return: (Pandas Series, Pandas DataFrame) respectively for the
                 percentage stats and the residuals series.
        """
        x1 = input
        x2 = output
        x1 = pd.concat([x1, x1.sum(axis=1)], axis=1)
        x2 = pd.concat([x2, x2.sum(axis=1)], axis=1)
        x1.columns = list(input.columns) + ['aggregate']
        x2.columns = list(output.columns) + ['aggregate']
        e_i = Metrics.E_apps(x1)
        e_o = Metrics.E_apps(x2)
        percents = 100 * e_o / e_i
        residuals = x1.sub(x2)
        return percents, residuals
