import numpy as np
import pandas as pd

"""
### V.0 ### 11/05/2020
- appliances are just modeled by a single value which is the power level. All appliances are single state appliances.
- appliances are added if the SSP problem's output does not match the aggreagted consumption.
- only one appliance can be set on at at time
"""


class Appliance():
    """
    Class to represent an appliance.

    :param clock: Integer. +1 for each step since the last change of state
    :param clok_log: List of Integers. Updated every change of state. If the appliance was on, we log the clock positively, negatively otherwise.
    :param sate: Boolean. True if appliance is on, False otherwise.
    :param power_level: Any number. Level of power of the single state appliance.

    """

    def __init__(self, dy):
        """
        Instanciate an appliance of power level dy.

        :param dy: Value of the power level of the appliance. Must be strictly positive.
        """
        assert dy > 0

        self.clock = 0
        self.clock_log = []

        self.state = False
        self.power_level = int(dy)

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
    :param power: Last value of power tracked
    :param n_app: Integer. Number of recorded appliances.
    :param state: Array of lenght nb_app with binary variables indicating the appliances states.
    :param power: Float. Last recorded power value by the tracker.
    """

    def __init__(self, T_detect=60, T_error=.05):
        """
        :param T_detect: Any number. Treshold above which transitions are taken into account. Default 60.
        :param T_error: Any number. Maximum error authorized before considering new app.
                        For instance T_error = .05 would mean that power is ecpected to matched ina 5 percent range.
        """
        self.T_error = T_error
        self.T_detect = T_detect

        self.state = []
        self.n_app = 0
        self.dictionary = []
        self.power = 0

    def add(self, dy):
        """
        Adds to the dictionnary an appliance of power level dy.

        :param dy: Ay number. Expeceted to be positive.
        """
        self.dictionary.append(Appliance(dy))
        self.n_app += 1

    def pass_time(self):
        """
        Increments the internal clock state of all appliances.
        """
        for app in self.dictionary:
            app.pass_time()

    def track(self, input):
        """
        Takes a signal as input and dissagregates it.

        :param input: Pandas Series. Signal to disaggregate.
        :return: Pandas DataFrame. One column for each tracked appliance.
        """
        dissagregated = []
        nb_app = []
        index = input.index

        for x in input:
            self.next_step(x)
            dissagregated.append(self.get_power_vector())

        tp_concat = [pd.Series(a) for a in dissagregated]
        result = pd.concat(tp_concat, axis=1).transpose().fillna(0)
        result = result[np.flip(result.columns)]
        result.index = index

        return result

    def next_step(self, y):
        """
        Reads new value y and run one step of the tracking algorithm.
        This design is inspired by https://doi.org/10.1109/APPEEC45492.2019.8994618.
        :param y: Any positive number. Power reading.
        """
        dy = y - self.power
        if dy > self.T_detect:
            error, mask = self.improved_bellman(y, self.T_error)

            if error < self.T_error:
                self.update_states(mask)

            else:
                self.add(dy)
                self.dictionary[-1].turn_on()

        elif dy < - self.T_detect:

            error, mask = self.improved_bellman(y, self.T_error)
            self.update_states(mask)

        self.pass_time()
        self.power = y

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

    def update_states(self, bools):
        """
        Update the states of each appliances given the array of boolean.

        :param bools: Array like. Booleans, True if app must be on, False otherwise.
        """
        for a, b in zip(self.dictionary, bools):
            if a.state and (not b):
                a.turn_off()
            elif not a.state and b:
                a.turn_on()

    def KP_greedy(self, y):
        """
        Greedy implementation of the knapsack problem. Depreciated.
        """

        X = pd.Series([a.power_level for a in self.dictionary], dtype='float64').sort_values()
        w_conso = 0
        result = [False] * self.n_app

        for k, conso in enumerate(X):
            if w_conso + conso < y:
                result[k] = True
                w_conso = w_conso + conso

        error = w_conso
        if y > 0:
            error = np.abs(w_conso - y) / y
        return error, result

    def improved_bellman(self, y, tol):
        """
        Implements improved Bellman algorithm for the Subset Sum Problem. Complexity is n_app*y in time, and n_app + y in space.

        :param y: Any positive number. Value to match.
        :param to: Float in (0., 1.). Percentage of tolerance to match the power value.
        :return: (Float, Arry of bools). Percentage of y that has been explained.
                 Booleans that indicate the corresponding appliances states.
        """

        c = int(y * (1 + tol))

        weights = np.array([(k, a.power_level) for k, a in enumerate(self.dictionary) if a.power_level < c])

        if len(weights) == 0:
            return c, [False] * self.n_app

        else:

            R = [0]
            r = [0] * (c + 1)

            for k in range(len(weights)):
                R_prim = [x + weights[k][1] for x in R if x + weights[k][1] < c]
                R_next = list(set().union(R, R_prim))
                for d in np.setdiff1d(R_next, R):
                    r[d] = k
                R = R_next
            total = max(R)
            error = np.abs(total - y) / y
            X_opt = np.array([False] * len(weights))
            while total != 0:
                X_opt[r[total]] = True
                total = total - weights[r[total]][1]
            mask = [False] * self.n_app
            for k, _ in weights[X_opt]:
                mask[k] = True

            return error, mask
