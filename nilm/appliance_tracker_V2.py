from utils.compute import Functions, Metrics
import numpy as np
import pandas as pd
from utils.NN_modules import data_manager, AppNet
import torch
import torch.nn as nn
from nilm.solvers import Solver
from sklearn.preprocessing import StandardScaler
"""
### V.2 ###
- appliances are modeled by :
    - A neural network that outputs the probability for the app to be on at a given time
    - a power value
- The framework  imitates a reinforcment learning task. Each appliance model is trained given a ground truth produced by a LP task.
- The input is made of several past power readings as well as the hour of the current one.
"""


class Appliance():
    """
    Class to represent an appliance.

    :param clock: Integer. +1 for each step since the last change of state
    :param clok_log: List of Integers. Updated every change of state. If the appliance was on, we log the clock positively, negatively otherwise.
    :param sate: Boolean. True if appliance is on, False otherwise.
    :param power_level: Any number. Level of power of the single state appliance.

    :param buffer: utils.NN_modules.data_manager. Holds the training samples.
    :param model: Neural network that outputs the probability for the appliance to be on.
    :param optimizer: Optimizer of the model.
    :param criterion: Loss used to train the model.
    :param learning_rate: Learning rate used for training.
    :param prediction_log: Array. Historical of all predictions made by the model.
    :param gt_log: Array. Historical of all the ground truth given by the environnement. (Ground truth is given by a LP problem.)

    :param observed_alone: Is true if the appliance has been observed alone.

    """
    def __init__(self, dy, input, buffer_size = 1440, train_window = 15):
        """
        Instanciate an appliance of power level dy.

        :param dy: Value of the power level of the appliance. Must be strictly positive.
        :param input: First input given to the model when the appliance is initialized.
        :param buffer_size: Integer. Size of the buffer from which the training samples are picked.
        :param train_window: Integer. Size of the past data given as input yo the odel. (besides the hour)
        """
        assert dy > 0

        self.train_window = train_window
        self.buffer = data_manager(df_size = buffer_size)
        self.model = AppNet(input_size = train_window + 2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.BCEWithLogitsLoss(reduction = 'sum')
        self.learning_rate = 1e-3

        self.prediction_log = []
        self.gt_log = []
        self.loss_log = []

        self.clock = 0
        self.clock_log = []

        self.state = False
        self.power_level = int(dy)
        self.observed_alone = False

        self.predict_proba(input)

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

    def predict_proba(self, line, log = True):
        """
        Outputs the prediction made by the model from the given input line.

        :param line: Array like of size train_window + 2. Input of the model.
        :param log: Boolean. If True, the output is logged.
        :return: Float. The estimated probability.
        """
        self.model.eval()
        sigmoid = torch.sigmoid
        pred = sigmoid(self.model(torch.FloatTensor([line]))).detach().numpy()[0][0]
        if log :
            self.prediction_log.append(pred)
        return pred

    def predict(self, line, log = False):
        """
        Outputs the decision made from the probality prediction. That is that if probality of the appliance is
        above .5, it is considered to be predicted as on.

        :param line: Array like of size train_window + 2. Input of the model.
        :param log: Boolean. If True, the output is logged.
        :return: 0 or 1.
        """
        proba = self.predict_proba(line, log = log)
        return (proba > .5)*1

    def pass_time(self):
        """
        Increments the clock value.
        """
        self.clock += 1

    def train(self, batch_size, n_batch):
        """
        Picks n_batches of size batch_size from the buffer and train the model on them.
        """
        if batch_size*n_batch > self.buffer.get_size() :
            train_loss = []
            self.model.train()
            for _ in range(n_batch):
                batch = self.buffer.get_random_batch(batch_size=batch_size)
                inputs = torch.FloatTensor(batch[:,:-1])
                targets = torch.FloatTensor(batch[:,-1]).reshape(-1,1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                l = loss.data.item()
                train_loss.append(l)
            self.loss_log.append(np.mean(train_loss))

    def add_data(self, input):
        """
        Adds the input to the buffer.
        """
        line = np.concatenate([input, np.array([float(self.state)])])
        if self.buffer.is_full() :
            self.buffer.dump(.1)
        self.buffer.add(line)

    def plot_metrics(self, window):
        """
        Evaluates on the binary classification standpoint the past predictions.

        :param window: Integer. The size of the batches on which the evaluation is made.
        :return: None.
        """
        N = len(self.gt_log)//window
        mets = []
        for k in range(N):
            a,_ = Metrics.evaluate(self.gt_log[k*window:(k+1)*window], np.array(self.prediction_log[k*window:(k+1)*window]) > .5)
            mets.append(a)
        tab = pd.concat(mets, axis = 1).transpose()
        tab.plot.line()


class Tracker():
    """
    Class that implements an appliance tracker.

    :param dictionary: List of recorded appliances.
    :param n_app: Integer. Number of recorded appliances.
    :param power: Float. Last recorded power value by the tracker.
    :param scaler: Sacler used for scaling the inputs before the go in the models.
    :param decision: Log of all decisions made - that is which decision process was used to make the predictions. See next_step().
    """

    def __init__(self, aggregated, T_detect=60, T_error=.05, train_window = 15, gamma_rkp = 10):
        """
        :param aggregated: Pandas Series. The aggregated signal on which the tracker is set.
        :param T_detect: Any number. Treshold above which transitions are taken into account. Default 60.
        :param T_error: Float within (0,1). Maximum error authorized before considering new app.
                        For instance T_error = .05 would mean that power is ecpected to matched in 5 percent range.
        :param train_window: Integer. Number of past data given as an input to the appliances model.
        :param gamma_rkp: Any positive number. Gamma parameter for the robust knapsack problem. Please refer to the solver guide for more info.
        """
        self.aggregated = aggregated
        self.scaler =  StandardScaler().fit(np.array(aggregated).reshape(-1,1))

        self.gamma_rkp = gamma_rkp

        self.T_error = T_error
        self.T_detect = T_detect
        self.train_window = train_window

        self.decision = []

        self.clock = 0 #time since instanciation

        self.dictionary = []

        self.power = 0

    def add_new_data(self, aggregated):
        """
        Replace param aggregated.
        """
        self.aggregated = aggregated

    @property
    def n_app(self):
        return len(self.dictionary)

    def add(self, dy, input):
        """
        Adds to the dictionnary an appliance of power level dy.

        :param dy: Ay number. Expeceted to be positive.
        :param input: The first training sample for the new appliance.
        """
        self.dictionary.append(Appliance(dy, input, train_window = self.train_window))

    def pass_time(self):
        """
        Increments the internal clock state of all appliances.
        """
        for app in self.dictionary:
            app.pass_time()
        self.clock += 1

    def train_models(self, batch_size, n_batch):
        """
        Trains all the individual models for n_batch of size batch_size.
        """
        for a in self.dictionary :
            a.train(batch_size, n_batch)

    def update_dictionnary(self):
        """
        Not used. Method to clean the appliance dictionary of models that did not learn well.
        """
        to_del = []
        for k, a in enumerate(self.dictionary):
            m,_ = Metrics.evaluate(a.gt_log, np.array(a.prediction_log) > .5)
            if np.mean(m) < tresh :
                to_del.append(k)

        for k in to_del[::-1]:
            del self.dictionary[k]

    def track(self, verbose=False, train_rate = 10, batch_size = 10, n_batch = 10):
        """
        Procede to track the signal self.aggregated.

        :param verbose: If one wants to have system messages printed.
        :param train_rate: Integer. Number of tracking step before the appliances models are trained.
        :param batch_size: Integer. Size of the batches for the training.
        :param n_batch: Integer. Number of batches at each training step.
        :return: Pandas DataFrame. One column for each tracked appliance.
        """
        dissagregated = []
        nb_app = []
        index = self.aggregated.index[self.train_window:]

        for k in range(self.train_window, self.aggregated.shape[0]):
            self.next_step(k, verbose=verbose)
            dissagregated.append(self.get_power_vector())
            if (k - self.train_window + 1 )%train_rate == 0:
                self.train_models(batch_size= batch_size, n_batch = n_batch)

        tp_concat = [pd.Series(a) for a in dissagregated]
        result = pd.concat(tp_concat, axis=1).transpose().fillna(0)
        result.index = index

        return result


    def next_step(self, k, pass_time=True, verbose=False):
        """
        Procede to the tracking algorithm applied on the k-th input data.

        :return: Array like. The binary vector of which appliance is said to be on.
        """
        y = self.aggregated.iloc[k]
        final = []

        input = self.make_input(np.array(self.aggregated.iloc[k-self.train_window+1:k+1]), self.aggregated.index[k])

        profits = [np.log(1+max(.01, a.predict_proba(input, log = pass_time))) for a in self.dictionary]
        weights = [a.power_level for a in self.dictionary]

        dy = y - self.power

        if dy > self.T_detect :

            if self.n_app > 0 :

                result_rkp, profit_rkp = Solver.RKP(weights = weights, delta_w = self.T_error, profits = profits, gamma = self.gamma_rkp, c = y*(1+self.T_error))
                y_pred_rkp = np.sum([x*w for x,w in zip(result_rkp, weights)])
                err_rkp = np.abs(y-y_pred_rkp)/y

                if err_rkp > self.T_error :

                    result_ssp, y_pred_ssp = Solver.RKP(weights = weights, profits = weights, delta_w = self.T_error, gamma = 0, c = y*(1+self.T_error))
                    err_ssp = np.abs(y-y_pred_ssp)/y

                    if err_ssp > self.T_error :

                        final = self.get_state() + [True]
                        self.add(dy, input)

                        self.decision.append(0)

                    else :
                        final = result_ssp

                        self.decision.append(1)

                else :
                    final = result_rkp

                    self.decision.append(2)

            else :
                final = [True]
                self.add(dy, input)

                self.decision.append(0)

            self.update_states(final, input, log = pass_time)

        elif dy < - self.T_detect :

            result_rkp, profit_rkp = Solver.RKP(weights = weights, delta_w = self.T_error, profits = profits, gamma = self.gamma_rkp, c = y*(1+self.T_error))
            y_pred_rkp = np.sum([x*w for x,w in zip(result_rkp, weights)])
            err_rkp = np.abs(y-y_pred_rkp)/y

            if err_rkp > self.T_error :

                result_ssp, y_pred_ssp = Solver.RKP(weights = weights, profits = weights, delta_w = self.T_error, gamma = 0, c = y*(1+self.T_error))
                err_ssp = np.abs(y-y_pred_ssp)/y

                if err_ssp > self.T_error :

                    self.update_states(result_ssp, input, log = False)
                    self.power = np.sum([w*x for w,x in zip(weights, result_ssp)])
                    final = self.next_step(k, pass_time = False)

                else :
                    final = result_ssp

                    self.decision.append(1)

            else :
                final = result_rkp

                self.decision.append(2)

            self.update_states(final, input, log = pass_time)

        else :
            final = self.get_state()
            self.update_states(final, input, log = pass_time)

            self.decision.append(0)

        if pass_time:
            self.pass_time()

        self.power = y

        return final

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
                vec.append(a.power_level)
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


    def update_states(self, bools, input, log):
        """
        Update the states of each appliances given the array of boolean.
        Also adds the correponding line to the appliances buffers.

        :param bools: Array like. Booleans, True if app must be on, False otherwise.
        :param input: Array like. Input for the models.
        :param log: Boolean. Wether or not the operation must be logged.
        """
        for a, b in zip(self.dictionary, bools):
            if log :
                a.gt_log.append(b)
            if a.state and (not b):
                a.turn_off()
            elif (not a.state) and b:
                a.turn_on()
            if log :
                a.add_data(input)

        if np.sum(self.get_state()) == 1 :
            self.dictionary[np.argmax(self.get_state())].observed_alone = True


    def make_input(self, data, timestamp):
        """
        Makes input for the models. The first self.train_window digits are the past power readings. The last are the hour given
        in a trigonometrical format.

        :param data: Array like. Past power readings.
        :timestamp: Datetime. Timestamp of the current power reading.
        """
        hour = Functions.convert_to_trig(timestamp)
        scaled = self.scaler.transform(np.array(data).reshape(-1,1)).reshape(1,-1)[0]
        line = np.concatenate([scaled, hour])
        return line
