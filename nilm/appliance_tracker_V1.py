from utils.compute import Functions
import numpy as np
import pandas as pd
from scipy.integrate import quad

"""
### V.1 ###
- appliances are modled by :
- distributions of time off (exponential) and time on (gaussian)
- a power value.
"""


class Appliance():
    """
    Class to represent an appliance.

    :param clock: Integer. +1 for each step since the last change of state
    :param clok_log: List of Integers. Updated every change of state. If the appliance was on, we log the clock positively, negatively otherwise.
    :param sate: Boolean. True if appliance is on, False otherwise.
    :param power_level: Any number. Level of power of the single state appliance.
    :param time_model_on: Function. Probability distribution of the span the appliance is on.
    :param time_model_off: Function. Probability distribution of the span the appliance is off.
    :param observed_alone: Boolean. True if this appliance has been tracked alone at least once.
    """

    def __init__(self, dy):
        """
        Instanciate an appliance of power level dy.

        :param dy: Value of the power level of the appliance. Must be strictly positive.
        """
        assert dy > 0

        self.clock = 0
        self.clock_log = []
        self.time_model_on = None
        self.time_model_off = None
        self.state = False
        self.power_level = int(dy)
        self.observed_alone = False

    def infer_distribs(self, min_sample=5):
        """
        Infer the off and on time distributions from clock log.

        :param min_sample: Integer. Minimum number of different spans of on and off times for a distribution to be infered.
        """
        on_times = np.array(
            [x for x in self.clock_log if x > 0]).reshape(-1, 1)
        off_times = np.array([np.abs(x)
                              for x in self.clock_log if x < 0]).reshape(-1, 1)
        if len(on_times) > min_sample and len(off_times) > min_sample:
            self.time_model_on = Functions.gaussian(mu=np.mean(
                on_times), sig=np.std(on_times), normalized=True)
            self.time_model_off = Functions.expo_law(lamb=np.mean(off_times))

    def get_proba_on(self):
        """
        From the time distributions, return the proba that the appliance next state is on.
        """

        if self.state:
            return quad(self.time_model_on, self.clock, self.clock + 2)[0]
        else:
            return .5
            # return 1 - quad(self.time_model_off, self.clock , self.clock + 2)[0]

    def has_distrib(self):
        """
        Returns True if the appliance has been assigned a probability distribution.
        """
        return self.time_model_on is not None

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
    :param power: Float. Last recorded power value by the tracker.
    :param clock: Integer. Time since the instanciation of the tracker.
    """

    def __init__(self, T_detect=60, T_error=.05, T_proba=.9):
        """
        :param T_detect: Any number. Treshold above which transitions are taken into account. Default 60.
        :param T_error: Float within (0,1). Maximum error authorized before considering new app.
                        For instance T_error = .05 would mean that power is ecpected to matched in 5 percent range.
        :param T_proba: Float within (0,1). Minimum ON - probability required for it be considered as relevant.
        """
        self.T_error = T_error
        self.T_detect = T_detect
        self.T_proba = T_proba
        self.clock = 0
        self.dictionary = []
        self.power = 0

    @property
    def n_app(self):
        return len(self.dictionary)

    def add(self, dy):
        """
        Adds to the dictionnary an appliance of power level dy.

        :param dy: Ay number. Expeceted to be positive.
        """
        self.dictionary.append(Appliance(dy))

    def pass_time(self):
        """
        Increments the internal clock state of all appliances.
        """
        for app in self.dictionary:
            app.pass_time()
        self.clock += 1

    def update_distribs(self):
        """
        Updates the appliances distributions. If the appliance has not been observed alone, it is deleted from the appliance dictionnary.
        Otherwise, if the appliance has been on for less than 10% of the frame, it is also deleted from the dictionnary.
        Otherwise, the new distributions of time-on and time-off models are updated relatively to the logs.
        """
        last_update = self.clock
        to_del = []
        for k, a in enumerate(self.dictionary):

            if not a.observed_alone:
                to_del.append(k)
            else:
                normed = np.array(a.clock_log) / \
                    np.sum([np.abs(x) for x in a.clock_log])
                if np.sum([x for x in normed if x > 0]) < .1:
                    if len(normed) < 6 or np.std([x for x in normed if x < 0]) > .1:
                        to_del.append(k)
                    else:
                        a.infer_distribs()
                else:
                    a.infer_distribs()

        for k in to_del[::-1]:
            del self.dictionary[k]

    def track(self, input, verbose=False):
        """
        Takes a signal as input and dissagregates it.

        :param input: Pandas Series. Signal to disaggregate.
        :return: Pandas DataFrame. One column for each tracked appliance.
        """
        dissagregated = []
        nb_app = []
        index = input.index

        for x in input:
            self.next_step(x, verbose=verbose)
            dissagregated.append(self.get_power_vector())

        tp_concat = [pd.Series(a) for a in dissagregated]
        result = pd.concat(tp_concat, axis=1).transpose().fillna(0)
        result.index = index

        return result

    def proba_available(self, apps):
        """
        Returns True if all appliances have been given probability distributions. Else return False.

        :param apps: Array like of Appliance. Appliances to be checked.
        """
        for app in apps:
            if not app.has_distrib():
                return False
        return True

    def next_step(self, y, pass_time=True, iter=0, verbose=False):
        """
        Reads new value y and run one step of the tracking algorithm.

        :param y: Any positive number. Power reading.
        """
        y = int(y)
        dy = y - self.power
        if dy > self.T_detect:

            error, mask = self.improved_bellman(y, self.T_error)

            if verbose:
                print("err {} prob {}".format(error, proba))
            if error < self.T_error:
                self.update_states(mask)

            else:
                self.add(dy)
                self.dictionary[-1].turn_on()

        elif dy < - self.T_detect:

            error, mask = self.improved_bellman(y, self.T_error)

            self.update_states(mask)

            if error > self.T_error:
                """
                Here we have a case of an appliance turning off but other yet untracked appliances have been turned on meanwhile.
                Thus we trick the tracker by doing another step during which the time do not pass.
                """
                self.power = np.sum(self.get_power_vector())
                self.next_step(y, pass_time=False, iter=iter + 1)

        if pass_time:
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
            elif (not a.state) and b:
                a.turn_on()
        if np.sum(self.get_state()) == 1:
            self.dictionary[np.argmax(self.get_state())].observed_alone = True

    def get_proba_decision(self, decision):
        """
        Returns the probability of a vector of bools given the time distribitions.
        """
        p = 1
        for a, bool in zip(self.dictionary, decision):
            w = np.sum([x for x in a.clock_log if x > 0]) / self.clock
            pk = 0.5
            if a.has_distrib():
                pk = a.get_proba_on()
                if not bool:
                    pk = (1 - pk)
            p = pk

        return p

    def improved_bellman(self, y, tol):
        """
        Implements improved Bellman algorithm for the Subset Sum Problem. Complexity is n_app*y in time, and n_app + y in space.

        :param y: Any positive number. Value to match.
        :param tol: Float in (0., 1.). Percentage of tolerance to match the power value.
        :return: (Float, Arry of bools). Percentage of y that has been explained.
                 Booleans that indicate the corresponding appliances states.
        """

        c = int(y * (1 + tol))

        appliances = [(k, a) for k, a in enumerate(
            self.dictionary) if a.power_level <= c]

        if len(appliances) == 0:
            return 1, [False] * self.n_app

        else:
            weights = np.array([a[1].power_level for a in appliances])
            R = [0]
            r = [0] * (c + 1)

            for k in range(len(weights)):
                R_prim = [x + weights[k]
                          for x in R if x + weights[k] < c]
                R_next = list(set().union(R, R_prim))
                for d in np.setdiff1d(R_next, R):
                    r[d] = k
                R = R_next

            values = [d for d in R if d > int(y * (1 - tol))]

            if len(values) == 0:
                return 1, [False] * self.n_app

            else:
                values.sort(key=lambda x: np.abs(x - y) / y)

                masks = []
                for va in values:
                    total = va
                    X_opt = []
                    while total != 0:
                        X_opt.append(r[total])
                        total = total - weights[r[total]]
                    mask = [False] * self.n_app
                    for k in X_opt:
                        mask[appliances[k][0]] = True
                    masks.append(mask)

                #probs = [self.get_proba_decision(mask) for mask in masks]
                nb_app = [np.sum(mask) for mask in masks]
                indx = np.argmin(nb_app)

                mask = masks[indx]
                total = values[indx]
                error = np.abs(total - y) / y

            return error, mask
