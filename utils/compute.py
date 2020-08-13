import numpy as np
import pandas as pd
from itertools import *
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import confusion_matrix


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
    def convert_to_trig(timestamp):
        theta = (timestamp.hour * 60 + timestamp.minute)*np.pi/(24*30)
        return np.array([np.cos(theta), np.sin(theta)])

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
            factor = 1 / (np.sqrt(2 * np.pi) * sig)

        def f(x):
            return factor * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        return f

    @staticmethod
    def resample(df, rate):
        """
        Depreciated.

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

    @staticmethod
    def merge_dict(dictionnaries):
        """
        Merge together the dictionnaries without duplicate, which must all have the same keys.

        :param dictionnaries: list of dictionnaries.
        :return: Dictionnary.
        """
        result = {}
        d1 = dictionnaries[0]
        for key, values in d1.items():
            new_values = values
            if type(values) != list:
                new_values = [new_values]
            for d in dictionnaries[1:]:
                to_add = d[key]
                if type(to_add) != list:
                    to_add = [to_add]
                new_values = new_values + list(set(to_add) - set(new_values))
            result[key] = new_values
        return result

    @staticmethod
    def get_window(serie, window_size):
        """
        Returns a random sample of the serie of size window_size and where the appliance is not always off.

        :param serie: Pandas Series. The data to sample from.
        :param window_size: Size of the sample.
        :return: Pandas Series. The selected window.
        """
        areas_of_interest = serie[serie > serie.mean()].index
        ref = np.where(serie.index == np.random.choice(
            areas_of_interest))[0][0]
        high = ref + window_size // 2
        low = ref - window_size // 2
        if high > serie.shape[0]:
            high = serie.shape[0]
            low = serie.shape[0] - window_size
        elif low < 0:
            low = 0
            high = window_size
        return serie.iloc[low:high]

    @staticmethod
    def iter_param(param_grid):
        """
        Provides an iterator over all possible parameter combinations.

        :param param_grid: Dictionnary.
        :return: Iterator.
        """

        items = sorted(param_grid.items())
        if not items:
            yield {}
        else:
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                yield params

    @staticmethod
    def infer_distrib(X, max_n_components, weight_tresh):
        model = BayesianGaussianMixture(n_components=max_n_components, weight_concentration_prior=.5,
                                        max_iter=1000, covariance_type='spherical', weight_concentration_prior_type='dirichlet_process')
        model.fit(X)
        while np.min(model.weights_) < weight_tresh:
            n_c = np.sum(model.weights_ > weight_tresh)
            model = BayesianGaussianMixture(n_components=n_c, weight_concentration_prior=.5, max_iter=1000,
                                            covariance_type='spherical', weight_concentration_prior_type='dirichlet_process')
            model.fit(X)
        return model

    @staticmethod
    def get_proba(model, x):
        """
        Returns the probability of having x given the distribution described by model, which must be
        a gaussian mixture model.
        """
        preds = model.predict_proba([[x]])
        regime = model.predict([[x]])[0]

        return np.sum(model.weights_ * preds)

    @staticmethod
    def expo_law(lamb):
        def f(x):
            return lamb * np.exp(-lamb * x)
        return f

    @staticmethod
    def jcoeff(rect1, rect2, y_tare = False):
        x_mn1, x_mx1, y_mn1, y_mx1 = rect1
        x_mn2, x_mx2, y_mn2, y_mx2 = rect2
        if y_tare :
            y_mx1 = y_mx1 - y_mn1
            y_mn1 = 0
            y_mx2 = y_mx2 - y_mn2
            y_mn2 = 0
        inter = max(0, min(x_mx1, x_mx2) - max(x_mn1, x_mn2))* max(0, min(y_mx1, y_mx2) - max(y_mn1, y_mn2))
        union = (y_mx1-y_mn1)*(x_mx1- x_mn1) + (y_mx2-y_mn2)*(x_mx2- x_mn2) - inter
        return inter/union

    @staticmethod
    def max_coef(rect1, rect2):
        x_mn1, x_mx1, y_mn1, y_mx1 = rect1
        x_mn2, x_mx2, y_mn2, y_mx2 = rect2

        y_mx1 = y_mx1 - y_mn1
        y_mn1 = 0
        y_mx2 = y_mx2 - y_mn2
        y_mn2 = 0
        inter = max(0, min(x_mx1, x_mx2) - max(x_mn1, x_mn2))* max(0, min(y_mx1, y_mx2) - max(y_mn1, y_mn2))
        area_1 = (x_mx1-x_mn1)*(y_mx1-y_mn1)
        return inter/area_1


class Metrics:
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

    @staticmethod
    def mse():
        """
        Returns a callable that implements the mse error.
        """
        def metric(input, output):
            """
            :param input: Pandas Series.
            :param output: Pandas Series.
            :return: Float. The mse between the two series.
            """
            a = (input - output)**2
            return a.mean()
        return metric

    @staticmethod
    def amape():
        """
        Returns a callable that impelments an adapted version of the mean absolute percentage error.
        """
        def metric(input, output):
            """
            :param input: Pandas Series.
            :param output: Pandas Series.
            :return: Float. The error is computed relatively to the input.
            """
            a = (input - output).abs() / (1 + input)
            return a.mean()
        return metric

    @staticmethod
    def evaluate(target, source):
        """
        Evaluates on all classic metrics the performances of a binary classification.
        """
        tn, fp, fn, tp = confusion_matrix(target, source).ravel()
        acc = (tp+tn)/(tn+fp+fn+tp)
        rec = tp/(tp+fn)
        pre = tp/(tp+fp)
        spe = tn/(tn+fp)
        f1 = (2*pre*rec)/(pre+rec)
        result = pd.Series([acc,rec,pre,spe,f1])
        result.index = ['Acc', 'Rec', 'Pre', 'Spe', 'F1']

        mat = pd.DataFrame(confusion_matrix(target, source))
        mat.columns = pd.MultiIndex(levels = [['Actual'],['Negatives','Positives']], codes = [[0,0],[0,1]])
        mat.index = pd.MultiIndex(levels = [['Prediction'],['Negatives','Positives']], codes = [[0,0],[0,1]])
        dfStyler = mat.style.set_properties(**{'text-align': 'center'})
        mat =dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])

        return result, mat
