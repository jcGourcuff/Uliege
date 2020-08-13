from dataManager.load import DataSets
from utils.compute import Functions, Metrics
import numpy as np
import pandas as pd
from utils.NN_modules import data_manager, AppNet
import torch
import torch.nn as nn
from nilm.solvers import Solver
from sklearn.preprocessing import StandardScaler
import copy
import time

"""
### V.3 ###
On this version, a neural network is shared by all the appliances. When it comes to adding or removing one
from the dictionnary, an output is dynamically added or removed from the network.
"""


class Appliance():
    """
    Class to represent an appliance.

    :param clock: Integer. +1 for each step since the last change of state
    :param clok_log: List of Integers. Updated every change of state. If the appliance was on, we log the clock positively, negatively otherwise.
    :param sate: Boolean. True if appliance is on, False otherwise.
    :param power_mean: Any number. Mean power of the single state appliance.
    :param power_std: Any positive number. Power level standard deviation of the single state appliance.
    :param observed_alone: Boolean. True if the appliance has been observed alone.
    :param prediction_log: Array like. Holds the historic of predictions for this appliance.
    :param gt_log: Array-like. Holds the historic of ground truths for this appliance.

    """

    def __init__(self, dy, std):
        """
        Instanciate an appliance of power level dy.

        :param dy: power_mean.
        :param std: power_std.
        """
        assert dy > 0

        self.prediction_log = []
        self.gt_log = []

        self.clock = 0
        self.clock_log = []

        self.state = False
        self.observed_alone = False

        self.power_mean = dy
        self.power_std = std

    def get_power_sample(self):
        """
        Returns an array of values sampled from the inffered power distribution of the appliance.
        """
        return np.array([self.power_mean - self.power_std, self.power_mean, self.power_mean + self.power_std])

    def turn_on(self):
        """
        Turns the appliance on. Reset internal clock.
        """
        self.reset_clock()
        self.state = True

    def turn_off(self):
        """
        Turns the appliance off. Reset internal clock.
        """
        self.reset_clock()
        self.state = False

    def reset_clock(self):
        """
        If the appliance was on, we log the clock positively, negatively otherwise.
        Clock set to 0.
        """
        if self.state:
            self.clock_log.append(self.clock)
        else:
            self.clock_log.append(-self.clock)

        self.clock = 0

    def pass_time(self):
        """
        Increments the clock value.
        """
        self.clock += 1


class Tracker():

    """
    Class that implements an appliance tracker.

    :param dictionary: List of dictionary of appliance.
    :param n_app: Integer. Number of recorded appliances.
    :param power: Float. Last recorded power value by the tracker.
    :param min_value: Minimum value within the aggregated signal.
    :param buffer: NN_modules.data_manager. Buffer to hold training data.
    :param model: Dynamic Appnet.
    :param optimizer: Model optimizer.
    :param loss_fn: Criterion used to train the model.
    :param learning_rate: Learning rate used to train the model.
    :param preds: Arry like to hold the predictions.
    """

    def __init__(self, aggregated, T_detect=60, T_error=.05, buffer_size=1440, train_window=15):
        """
        :param aggregated: Pandas Series. The aggregated signal on which the tracker is set.
        :param T_detect: Any number. Treshold above which transitions are taken into account. Default 60.
        :param T_error: Float within (0,1). Maximum error authorized before considering new app.
                        For instance T_error = .05 would mean that power is ecpected to matched in 5 percent range.
        :param train_window: Integer. Number of past data given as an input to the appliances model.
        :param buffer_size: Integer. Size of the data buffer.
        """
        self.aggregated = aggregated

        self.min_value = min(aggregated)

        self.buffer_size = buffer_size
        self.buffer = data_manager(df_size=buffer_size)
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        self.learning_rate = 1e-3

        self.preds = []

        self.T_error = T_error
        self.T_detect = T_detect
        self.train_window = train_window

        self.clock = 0  # time since instanciation

        self.dictionary = []

        self.power = 0

    def add_new_data(self, aggregated):
        """
        Redefine self.aggregated.
        """
        self.aggregated = aggregated

    @property
    def n_app(self):
        return len(self.dictionary)

    def fit_scaler(self, data):
        """
        Fits a scaler on data.
        """
        self.scaler = StandardScaler().fit(np.array(data).reshape(-1, 1))

    def add_output(self):
        """
        Adds an output to the model.
        """
        if self.n_app == 0:
            self.model = AppNet(self.train_window)
        else:
            self.model.eval()
            new = AppNet(self.train_window)
            new.common_layers = copy.deepcopy(self.model.common_layers)
            new.fc = copy.deepcopy(self.model.fc)

            with torch.no_grad():
                last_layer = nn.Linear(
                    self.model.fc[5].weight.shape[1], self.model.fc[5].weight.shape[0] + 1)
                last_layer.weight[:-1] = self.model.fc[5].weight
                last_layer.bias[:-1] = self.model.fc[5].bias
                new.fc[5] = last_layer

            del self.model
            self.model = new
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.buffer.update_data(option='add')
        self.preds = [x + [0] for x in self.preds]

    def remove_output(self, k):
        """
        Remove the k-th output of the model.
        """
        self.model.eval()
        new = AppNet(self.train_window)
        new.common_layers = copy.deepcopy(self.model.common_layers)
        new.fc = copy.deepcopy(self.model.fc)

        with torch.no_grad():
            last_layer = nn.Linear(
                self.model.fc[5].weight.shape[1], self.model.fc[5].weight.shape[0] - 1)
            last_layer.weight[:] = torch.cat(
                [self.model.fc[5].weight[0:k], self.model.fc[5].weight[k + 1:]])
            last_layer.bias[:] = torch.cat(
                [self.model.fc[5].bias[0:k], self.model.fc[5].bias[k + 1:]])
            new.fc[5] = last_layer

        del self.model
        self.model = new
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.buffer.update_data(option='remove', index=k)
        self.preds = [x[:k] + x[k + 1:] for x in self.preds]

    def train_model(self, batch_size, n_train_loop):
        """
        Trains the model n_train_loop times on batches of size batch_size.
        """
        start_time = time.time()

        train_loss = []

        if self.buffer.get_size() > batch_size:

            self.model.train()

            for _ in range(min(n_train_loop, self.buffer.get_size() // batch_size)):

                batch = self.buffer.get_random_batch(batch_size=batch_size)
                inputs = torch.FloatTensor(batch[:, :-1].tolist())
                targets = torch.FloatTensor(batch[:, -1].tolist())
                outputs = self.model(inputs)

                loss = self.loss_fn(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                l = loss.data.item()
                train_loss.append(l)

        #print("Training model - {} seconds -".format(time.time() - start_time))
        return train_loss

    def add(self, dy):
        """
        Adds to the dictionnary an appliance of power level dy.

        :param dy: Ay number. Expeceted to be positive.
        """
        self.add_output()
        self.dictionary.append(Appliance(dy, std=dy * .05))

    def pass_time(self):
        """
        Increments the internal clock state of all appliances.
        """
        for app in self.dictionary:
            app.pass_time()
        self.clock += 1

    def update_dictionnary(self):
        """
        Not used in code. Method to remove appliances based on how well the model performs on predictions for it.
        """
        to_del = []
        for k, a in enumerate(self.dictionary):
            m, _ = Metrics.evaluate(a.gt_log, np.array(a.prediction_log) > .5)
            if np.mean(m) < tresh:
                to_del.append(k)

        for k in to_del[::-1]:
            del self.dictionary[k]

    def track(self, verbose=False, train_rate=10, batch_size=10, n_batch=10, matching_window=60, new_app_bound=1, power_sample_size=10):
        """
        Takes a signal as input and dissagregates it.

        :param verbose: True to display debugging messages.
        :param train_rate: Integer. Number of power value reading before the model is trained.
        :param batch_size: Integer. Size of the batches used for training.
        :param n_batch: Integer. Number of batches the model is trained on.
        :param matching_window: See Solver.KP_with_constraint.
        :param new_app_bound: Integer. See parameter M from Solver.KP_with_constraint.
        :power_sample_size: Number of power sample per appliance for the multiple knpasack problem.
        :return: Pandas DataFrame. One column for each tracked appliance.
        """
        if verbose :
            print("tracking...", end='\n')
        index = self.aggregated.index[self.train_window:]
        dissagregated = []

        n_to_track = self.aggregated.shape[0] - self.train_window + 1

        for k in range(self.train_window, self.aggregated.shape[0]):
            track_pct = np.round(100 * k / n_to_track, decimals=2)
            #print("{}%...".format(track_pct), end = '\r')

            power_sample_size = min(50, self.n_app * 10)
            self.next_step(k, matching_window, new_app_bound,
                           power_sample_size, verbose=verbose)
            dissagregated.append(self.get_power_vector())
            if (k - self.train_window + 1) % train_rate == 0:
                self.train_model(batch_size=batch_size, n_train_loop=n_batch)

        tp_concat = [pd.Series(a) for a in dissagregated]
        result = pd.concat(tp_concat, axis=1).transpose().fillna(0)
        result.index = index

        return result

    def MCKP(self, input_model, c, power_sample_size, matching_window, new_app_bound, nominal_weights=None, verbose=False):
        """
        Multiple Knapsack Problem. The methods is a variant from the KP problem where it is solved for multiple nominal values for each entity.

        :param input_model: Input for the model to outputs the profits of the KP problemes.
        :param c: Power value. Upper bound for the KP problem.
        :param power_sample_size: Number of power sample per appliance for the multiple knpasack problem.
        :param matching_window: See Solver.KP_with_constraint.
        :param new_app_bound: Integer. See parameter M from Solver.KP_with_constraint.
        :nominal_weights: If not none, a single KP problem is solved using these values as weights.
        :return: Tuple. The predicted states for the appliances, the actual prediction of the model, the weights retained after the mckp.

        """
        start_time = time.time()

        self.model.eval()
        profits = torch.sigmoid(self.model(
            torch.FloatTensor([input_model]))).detach().numpy()[0]
        weights = np.array([a.get_power_sample().tolist()
                            for a in self.dictionary]).transpose()
        if verbose :
            print(weights)
            print(profits)
        if nominal_weights is not None:
            profits = nominal_weights
            weights = [nominal_weights]

        negate_previous_states = ~self.compute_previous_states(
            matching_window=matching_window)

        KP_sols = [Solver.KP_with_constraint(
            w, profits, c, negate_previous_states, new_app_bound) for w in weights]

        index_best = min(range(len(KP_sols)),
                         key=lambda i: np.abs(KP_sols[i][1] - c))
        retained_weights = weights[index_best]
        X_pred, y_pred = KP_sols[index_best]

        if verbose:
            print("GT : {} | PRED : {} | NOM {}".format(
                c, y_pred, nominal_weights is not None))

        return X_pred, y_pred, retained_weights

    def next_step(self, k, matching_window, new_app_bound, power_sample_size, verbose=False, pass_time=True):
        """
        Reads new value y and run one step of the tracking algorithm.
        """
        y = self.aggregated.iloc[k] - self.min_value + 0.01

        x_final = []

        input = self.make_input(
            np.array(self.aggregated.iloc[k - self.train_window + 1:k + 1]))

        dy = y - self.power

        if dy > self.T_detect:

            if self.n_app > 0:

                x_pred, y_pred, w_pred = self.MCKP(
                    input, y, power_sample_size, matching_window, new_app_bound, verbose=verbose)

                error = np.abs(y_pred - y) / y

                if error < self.T_error:

                    x_final = x_pred

                else:

                    nominal_weights = [a.power_mean for a in self.dictionary]

                    x_ssp, y_ssp, w_ssp = self.MCKP(
                        input, y, power_sample_size, matching_window, new_app_bound, nominal_weights=nominal_weights, verbose=verbose)

                    error_ssp = np.abs(y_pred - y) / y

                    if error_ssp < self.T_error:

                        x_final = x_ssp

                    else:

                        x_final = self.get_state() + [1]
                        self.add(dy)

            else:
                x_final = [1]
                self.add(dy)

        elif dy < - self.T_detect:

            x_pred, y_pred, w_pred = self.MCKP(
                input, y, power_sample_size, matching_window, new_app_bound, verbose=verbose)

            error = np.abs(y_pred - y) / y

            if error < self.T_error:

                x_final = x_pred

            else:

                nominal_weights = [a.power_mean for a in self.dictionary]

                x_ssp, y_ssp, w_ssp = self.MCKP(
                    input, y, power_sample_size, matching_window, new_app_bound, nominal_weights=nominal_weights, verbose=verbose)

                error_ssp = np.abs(y_pred - y) / y

                if error_ssp < self.T_error:
                    x_final = x_ssp

                else:

                    self.power = y
                    self.update_states(x_ssp)
                    x_final = self.next_step(
                        k, matching_window, new_app_bound, power_sample_size, verbose=False, pass_time=False)

        else:

            x_final = self.get_state()

        if pass_time:

            self.make_n_save_data(input, x_final)
            self.update_states(x_final)
            self.preds.append(x_final)
            self.pass_time()

        self.power = y

        return x_final

    def get_power_vector(self, option='default'):
        """
        Returns a vector of length nb_app. Each paramater is set to the power
        value if the corresponding appliance is on, 0 otherwise.

        :param option: String. 'default' or 'negate'. 'default' corresponds to
                                behavior described above. If 'negate', off appliances numbers ar set to
                                the power level, and the on appliances are set to 0.
        :return: list. The power vector.
        """
        vec = []
        for a in self.dictionary:
            bool = a.state
            if option == 'negate':
                bool = not bool
            if bool:
                vec.append(a.power_mean)
            else:
                vec.append(0)
        return vec

    def get_state(self, option='default'):
        """
        Returns boolean array with True if the corresponding appliance is on.

        :param option: String. 'default' or 'negate'. 'default' corresponds to
                                behavior described above. If 'negate', the off
                                applinaces are set to True and on appliances to off.
        """
        vec = []
        for a in self.dictionary:
            bool = a.state
            if option == 'negate':
                bool = not bool
            vec.append(bool)
        return vec

    def update_states(self, bools):
        """
        Update the states of each appliances given the array of boolean.

        :param bools: Array like. Booleans, True if app must be on, False otherwise.
        """
        for a, b in zip(self.dictionary, bools):

            if a.state and (not b):
                a.turn_off()
            elif (not a.state) and b:
                a.turn_on()

        if np.sum(self.get_state()) == 1:
            self.dictionary[np.argmax(self.get_state())].observed_alone = True

    def make_input(self, data):
        """
        Builds and scale a formated input for the model.
        """
        line = self.scaler.transform(
            np.array(data).reshape(-1, 1)).reshape(1, -1)[0]
        return line

    def make_n_save_data(self, input, output):
        """
        Fills the buffer.
        """
        line = np.array(input.tolist() + [output])
        if self.buffer.is_full():
            self.buffer.dump(.1)
        self.buffer.add(line)

    def compute_previous_states(self, matching_window):
        """
        Computes which appliances within the previous frames, in the range of matching_window.
        Returns the negated vector.
        """
        if len(self.preds) > 0:
            previous_states = [self.preds[-1]]
            k = 2
            while k <= min(len(self.preds) - 1, matching_window):
                previous_states.append(self.preds[k])
                k += 1

            represented_apps = previous_states[0]
            for s in previous_states[1:]:
                represented_apps = np.logical_or(represented_apps, s)
            return represented_apps

        else:
            return np.array([True] * self.n_app)
