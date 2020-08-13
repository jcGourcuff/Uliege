
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilm.filters import Filters
from utils.compute import Metrics
from utils.compute import Functions
import copy
from nilm.solvers import Solver
from dataManager.load import DataSets
import scipy.misc
from PIL import Image
from datetime import datetime, timedelta
from scipy.special import comb
import itertools
import time
classes = {'residuals': 0, 'oven': 1, 'refrigerator': 2, 'dishwasher': 3, 'kitchen_outlets': 4,
           'lighting': 5, 'washer_dryer': 6, 'microwave': 7, 'bathroom_gfi': 8}
colors = {'residuals': 'black',
          'oven': 'blue',
          'refrigerator': 'orange',
          'dishwasher': 'green',
          'kitchen_outlets': 'red',
          'lighting': 'purple',
          'washer_dryer': 'brown',
          'microwave': 'pink',
          'bathroom_gfi': 'gray'}

color_codes = {'residuals': [0, 0, 0],
               'oven': [0, 0, 2],
               'refrigerator': [255, 145, 0],
               'dishwasher': [0, 255, 0],
               'kitchen_outlets': [255, 0, 0],
               'lighting': [196, 0, 255],
               'washer_dryer': [102, 51, 0],
               'microwave': [255, 153, 255],
               'bathroom_gfi': [160, 160, 160]}


"""
MAJOR ISSUES : code do not generlize well to other window_size/ sampling rate
"""

Window_size = 1440
Cons_Mx = 6000
Cons_Mn = 0
y_lim = 6000

columns_drops = {'1': ['oven_1', 'kitchen_outlets_1',
                       'washer_dryer_2', 'electric_heat_1', 'stove_1']}


class LabeledImagesMaker:
    """
    This class implements a set of methods to transform a numerical load disagregation data set
    into an image based one.
    """

    def __init__(self):
        pass

    @staticmethod
    def make_img(data):
        """
        Builds and return as a numpy array the image corresponding to the the 1D input data.
        """
        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        fig.set_size_inches((Window_size / 72, Cons_Mx / 72))
        ax.plot(np.arange(Window_size), data, c="black")
        ax.set_xlim(0, Window_size)
        ax.set_ylim(0, Cons_Mx)
        ax.axis('off')
        fig.add_axes(ax)
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer._renderer)
        plt.close()
        return np.flip(X[:, :, :3], axis=0)

    @staticmethod
    def plot(im, interp=False):
        """
        Plots an image given as an argument.
        """
        f = plt.figure(figsize=(20, 8), frameon=True)
        plt.imshow(im, interpolation=None if interp else 'none',
                   aspect='auto', cmap='CMRmap')
        plt.ylim((0, y_lim))
        plt.show()

    @staticmethod
    def draw_horizontal(img, x_min, x_max, y, thickness=3, color=[255, 0, 0], copy_=True):
        """
        Builds a new image where an horizontal line is drawn on img, from x_min to x_max at height y.
        :return: Numpy array. The new image.
        """
        new = img
        if copy_:
            new = copy.copy(img)

        if type(y) == list:
            for k in y:
                new[k, x_min:x_max +
                    1] = np.array([color] * (x_max - x_min + 1))
        else:
            for k in range(y, min(y + thickness, Cons_Mx - 1)):
                new[k, x_min:x_max +
                    1] = np.array([color] * (x_max - x_min + 1))
        return new

    @staticmethod
    def draw_vertical(img, x, y_min, y_max, thickness=3, color=[255, 0, 0], copy_=True):
        """
        Builds a new image where a vertical line is drawn on img, from y_min to y_max at absciss x.
        :return: Numpy array. The new image.
        """
        new = img
        if copy_:
            new = copy.copy(img)

        if type(x) == list:
            for k in x:
                new[y_min:y_max + 1,
                    k] = np.array([color] * (y_max - y_min + 1))
        else:
            for k in range(x, min(x + thickness, Window_size - 1)):
                new[y_min:y_max + 1,
                    k] = np.array([color] * (y_max - y_min + 1))
        return new

    @staticmethod
    def draw_boxes(img, coords, thickness=3, color=[255, 0, 0]):
        """
        Builds a new image where an empty square is drawn on img, whith vertices defined by coords.
        :return: Numpy array. The new image.
        """
        new = copy.copy(img)
        for k, box in enumerate(coords):
            c = color
            if type(c[0]) == list:
                c = color[k]

            x_min, x_max, y_min, y_max = box
            new = LabeledImagesMaker.draw_horizontal(
                new, x_min, x_max, y_min, thickness, c, copy_=False)
            new = LabeledImagesMaker.draw_horizontal(
                new, x_min, x_max, y_max, thickness, c, copy_=False)
            new = LabeledImagesMaker.draw_vertical(
                new, x_min, y_min, y_max, thickness, c, copy_=False)
            new = LabeledImagesMaker.draw_vertical(
                new, x_max, y_min, y_max, thickness, c, copy_=False)
        return new

    @staticmethod
    def get_max_height(signal, x_min, x_max, res=2):
        """
        Return y_high of bigest fitting rectangle between x_min and x_max included
        The found upper limit must be within x_min + res, x_max - res
        """
        wind = signal[min(x_min + res, x_max + 1 - res)                      :max(x_max + 1 - res, x_min + res)]
        if len(wind) == 0:
            return -1
        else:
            return min(wind)

    @staticmethod
    def search_vert_bounds(signal, x_min, x_max, y, res_y=4):
        """
        Search and returns the indexes between x_min and x_max for which there is a vertical line above y.
        """
        indxs = []
        curs = x_min

        while curs <= x_max:

            if signal[curs] > y + res_y:
                x = curs
                if len(indxs) == 0 and x_min == 0 and signal[0] > y + res_y:
                    indxs.append((0, x))
                else:

                    while curs <= x_max and signal[curs] > y + res_y:
                        curs += 1
                    indxs.append((x, curs - 1))

            curs += 1

        return indxs

    @staticmethod
    def draw_boundaries(signal, res_x=4, res_y=5):
        """
        Returns the set of boxes that fit the signal in the forms of a list of coordinates.
        """
        coords = []
        y_low = 0
        y_high = LabeledImagesMaker.get_max_height(signal, 0, Window_size - 1)
        coords.append([0, Window_size - 1, 0, y_high])
        y_low = y_high + 1
        search_areas = [(0, Window_size - 1, y_low)]

        while len(search_areas) > 0:
            x_min, x_max, y_low = search_areas[0]
            search_areas = search_areas[1:]

            vert_bounds = LabeledImagesMaker.search_vert_bounds(
                signal, x_min, x_max, y_low, res_y=res_y)

            for bs in vert_bounds:

                if bs[1] - bs[0] > res_x:

                    y_high = LabeledImagesMaker.get_max_height(
                        signal, bs[0], bs[1], res=res_x)

                    if y_high > -1:
                        coords.append([bs[0], bs[1], y_low, y_high])
                        search_areas.append((bs[0], bs[1], y_high + 1))
        return np.array(coords).astype(int)

    @staticmethod
    def make_boxes(serie, res_y=5, res_x=2, flatten_tresh=10, median_filter_kernel_size=3):
        """
        Apply filters to the signal and call draw_boundaries.
        """
        filtered = Filters.median_filter(
            serie, kernel_size=median_filter_kernel_size)
        filtered = Filters.flatten(filtered, treshold=flatten_tresh)
        boxes = LabeledImagesMaker.draw_boundaries(
            filtered, res_x=res_x, res_y=res_y)

        return boxes

    @staticmethod
    def make_solver_inputs(unlabeledBoxes, df):
        """
        Compute all the inputs for Solver.label_assignement from the set of boxes to label and the data there were built from.
        """
        N_app = df.shape[1]
        N_u = len(unlabeledBoxes)
        A_unlabeled = [(x[1] - x[0]) * (x[3] - x[2]) for x in unlabeledBoxes]

        N_apps = []
        P = []
        A_labeled = []
        Max_W = [[]] * N_u
        jacq_coefs = [[]] * N_u

        apps_with_boxes = []
        target_budgets = []

        for name in df.columns:
            boxes = LabeledImagesMaker.make_boxes(df[name])[1:]
            if boxes.shape[0] > 0:
                N_apps.append(boxes.shape[0])
                apps_with_boxes.append(name)
                target_budgets.append(
                    df[name].sum() - df[name].min() * Window_size)
                areas = [(x[1] - x[0]) * (x[3] - x[2]) for x in boxes]
                A_labeled = A_labeled + areas
                areas = np.array(areas) / np.sum(areas)
                P = P + areas.tolist()

                for k in range(N_u):
                    Max_W[k] = Max_W[k] + \
                        [Functions.max_coef(box, unlabeledBoxes[k])
                         for box in boxes]
                    jacq_coefs[k] = jacq_coefs[k] + \
                        [Functions.jcoeff(box, unlabeledBoxes[k])
                         for box in boxes]
            else:
                N_app -= 1

        return N_app, N_u, N_apps, P, A_labeled, A_unlabeled, Max_W, jacq_coefs, apps_with_boxes, target_budgets

    @staticmethod
    def compute_boxes_and_labels(data_set, low, demo=False, verbose=False, compute_image=True):
        """
        From data in data_set sliced from low to low + window_size, return the image of the raw signal,
        the set of boxes fitting the signal and the set of corresponding labels.
        """
        agg = data_set[low:low + Window_size].sum(axis=1)

        if verbose:
            print("computing unlabeled boxes...", end='\r')
        unlabeld_boxes = LabeledImagesMaker.make_boxes(agg)

        if len(unlabeld_boxes) > 1:

            img = -1
            if compute_image:
                fil = Filters.median_filter(agg, kernel_size=3)
                fil = Filters.flatten(fil, treshold=10)
                img = LabeledImagesMaker.make_img(fil)

            if verbose:
                print('computing solver inputs...', end='\r')
            N_app_, N_u_, N_apps_, P_, A_labeled_, A_unlabeled_, Max_W_, jacq_coefs_, apps_with_boxes_, target_budgets_ = LabeledImagesMaker.make_solver_inputs(
                unlabeld_boxes[1:], data_set[low:low + Window_size])

            if len(A_labeled_) > 0:
                if verbose:
                    print('solving...', end='\r')

                argmax_labels = Solver.label_assignement(
                    N_app_, N_u_, N_apps_, P_, A_labeled_, A_unlabeled_, Max_W_)
                argmax_labels_names = [apps_with_boxes_[k][:-2]
                                       for k in argmax_labels]
                if demo:
                    argmax_labels_names = [apps_with_boxes_[k]
                                           for k in argmax_labels]

                argmax_labels_names = ['residuals'] + argmax_labels_names

                return img, np.array(unlabeld_boxes), np.array(argmax_labels_names)
            else:
                return -1, [], []
        else:
            return -1, [], []

    @staticmethod
    def plot_unlabeled_on_image(img, unlabeld_boxes, color=[255, 0, 0]):
        """
        Plots and return the image of unlabeld_boxes drawn on img.
        """
        cop = copy.copy(img)
        cop = LabeledImagesMaker.draw_boxes(cop, unlabeld_boxes, color=color)
        LabeledImagesMaker.plot(cop)
        return cop

    @staticmethod
    def plot_boxes_with_labels(img, unlabeld_boxes, labs):
        """
        Plots and return the image of unlabeld_boxes drawn on img with labels.
        """
        lab_img = copy.copy(img)

        for box, lab in zip(unlabeld_boxes, labs):
            lab_img = LabeledImagesMaker.draw_boxes(
                lab_img, [box], color=color_codes[lab])
        LabeledImagesMaker.plot(lab_img)
        return lab_img

    @staticmethod
    def save_image(img, directory, name):
        """
        Save img in png format in directory with name name.
        """
        im = np.flip(img, axis=0)
        im = Image.fromarray(im[-y_lim:, :, :])
        im.save(directory + name + '.png')

    @staticmethod
    def evaluate_E_captured(signal, boxes):
        """
        Computes the ration between the area of the boxes over the area of the signal.
        """
        E_boxes = np.sum([(x[1] - x[0]) * (x[3] - x[2]) * 60 for x in boxes])
        E_signal = np.sum(signal) * 60
        return E_boxes / E_signal

    @staticmethod
    def max_dissagregation_quality(data_set, demo=False, verbose=False):
        """
        Depreciated.
        """
        nb_iteration = data_set.shape[0] // Window_size
        results = []
        col_names = []
        for k in range(nb_iteration):
            if verbose:
                print("preprocessing image n°{} out of {}".format(
                    k + 1, nb_iteration))
            _, u_boxes, labels = LabeledImagesMaker.compute_boxes_and_labels(
                data_set, k * Window_size, demo=demo)
            if len(labels) > 0:
                line = []
                for i in range(data_set.shape[1]):
                    name = data_set.columns[i][:-2]
                    if demo:
                        name = data_set.columns[i]
                    if name not in col_names:
                        col_names.append(name)
                    if np.sum(labels == name) > 0:
                        line.append(LabeledImagesMaker.evaluate_E_captured(
                            data_set.iloc[k * Window_size:(k + 1) * Window_size, i], u_boxes[labels == name]))
                    else:
                        line.append(-1)
                results.append(line)
                if verbose:
                    print('OK')
            else:
                if verbose:
                    print("Empty sequence")

        results = pd.DataFrame(np.array(results))
        results.columns = col_names
        return results

    @staticmethod
    def get_start_day_indexes(data_set):
        """
        Returns all indexes in data_set corresponding to 00:00.
        """
        indxs = np.where((dataset.index.hour == 0) &
                         (dataset.index.minute == 0))[0]
        if indxs[-1] + Window_size > dataset.shape[0]:
            indxs = indxs[:-1]

        return indxs

    @staticmethod
    def make_labeled_images(data, low_index, start_id=1, demo=False):
        """
        Computes all the input/outputs of the frame of data between low_index and low_index + window_size (that is as much as there are boxes in an image).
        """
        id = start_id
        img_list = []
        label_list = []
        img, unlabeledBoxes, labels = LabeledImagesMaker.compute_boxes_and_labels(
            data, low_index, demo=demo)
        unlabeledBoxes = unlabeledBoxes.tolist()
        if len(labels) > 0:
            for k, (box, lab) in enumerate(zip(unlabeledBoxes, labels)):
                black_boxes = np.array(
                    unlabeledBoxes[:k] + unlabeledBoxes[k + 1:])
                im = LabeledImagesMaker.draw_boxes(
                    img, black_boxes, color=[0, 0, 0])
                im = LabeledImagesMaker.draw_boxes(im, [box])
                img_list.append((id, im))
                area = LabeledImagesMaker.get_area(box)
                label_list.append((id, lab, area))
                id += 1
        return img_list, label_list, id

    @staticmethod
    def make_data_set(data, directory, label_num, start_id=1, demo=False):
        """
        Builds all images and labels csvs from data and save in directory.
        """
        indxs = LabeledImagesMaker.get_start_day_indexes(data)
        list_labs = []
        id_tracker = start_id
        for k_low in indxs:

            ims, labs, id_tracker = LabeledImagesMaker.make_labeled_images(
                data, k_low, start_id=id_tracker, emo=demo)

            list_labs = list_labs + labs
            for im_ in ims:
                id, im = im_
                LabeledImagesMaker.save_image(
                    im, directory + 'images/', str(id))
            del ims

        list_labs = pd.DataFrame(np.array(list_labs))
        list_labs.to_csv(directory + 'labels_' + str(label_num) +
                         '.csv', header=['id', 'label', 'area'], index=False)
        print('Last_index : {}'.format(id_tracker - 1))
        print('{} images saved'.format(list_labs.shape[0]))
        del list_labs
        return (id_tracker - 1)

    @staticmethod
    def get_area(box):
        """
        Returns the area of the box defined by the coordinates given as input.
        """
        x_max, x_min, y_max, y_min = box
        return (x_max - x_min) * (y_max - y_min)


class ArtificialSetGenerator:
    """
    This class implements a set of methods to generate from a REDD data set a set of new data sets.
    The data datasets are distinguished by wich and how many appliances are in it.
    """

    def __init__(self, data):
        self.df = data
        self.days_indexes = self.get_start_day_indexes()

    def get_start_day_indexes(self):
        """
        Returns all indexes in data_set corresponding to 00:00.
        """
        indxs = np.where((self.df.index.hour == 0) &
                         (self.df.index.minute == 0))[0]
        if indxs[-1] + Window_size > self.df.shape[0]:
            indxs = indxs[:-1]

        return indxs

    def find_interest_zones(self, app_names):
        """
        Returns for each appliance in app_names the indexes where these same appliances are on.
        """
        result = []
        for name in app_names:
            app_res = []
            for idx in self.days_indexes:
                if len(LabeledImagesMaker.make_boxes(self.df[name][idx:idx + Window_size])) > 1:
                    app_res.append(idx)
            result.append(app_res)
            if len(app_res) == 0:
                print(name)
        return result

    def translate_sample(self, serie, shift):
        """
        Shifts the serie by shift indexes.
        """
        idx = serie.index
        new_serie = pd.Series(np.roll(serie.values.reshape(1, -1)[0], shift))
        new_serie.index = idx
        return new_serie

    def get_shift_value(self, nb_sample):
        """
        Return nb_sample shift values drawn from a normal distribution.
        """
        return [int(x) for x in np.random.normal(0, Window_size // 3, nb_sample)]

    def make_custom_dfs_from_set_of_column_name(self, app_names):
        """
        Return a new data set made of the appliances in app_names.
        """
        interest_zones = self.find_interest_zones(app_names)
        factors_table = ArtificialSetGenerator.get_factors()
        for k, x in enumerate(interest_zones):
            if len(x) > factors_table.loc[len(app_names), 'nb_index_per_class']:
                interest_zones[k] = np.random.choice(
                    x, int(factors_table.loc[len(app_names), 'nb_index_per_class']))
        indxs_combinations = np.array(np.meshgrid(
            *interest_zones)).T.reshape(-1, len(app_names))

        days = []
        start_date = datetime(2000, 1, 1)
        for indxs in indxs_combinations:
            sub_dfs = [self.df[name][idx:idx + Window_size].reset_index(
                drop=True) for idx, name in zip(indxs, app_names)]
            shift_values = self.get_shift_value(len(sub_dfs))
            sub_dfs = [self.translate_sample(df, shift)
                       for df, shift in zip(sub_dfs, shift_values)]
            new_df = pd.concat(sub_dfs, ignore_index=True, axis=1)
            new_df.index = pd.date_range(
                start=start_date, periods=1440, freq='1min', normalize=True)
            new_df.columns = app_names
            start_date = start_date + timedelta(days=1)
            days.append(new_df)

        result = pd.concat(days)

        return result

    def represented_classes(self, data_set=None):
        """
        Returns the classes that are within the data.
        """
        pool = self.df.columns
        if type(data_set) == int:
            pool = pool.drop(columns_drops[str(data_set)])
        res = []
        for app in pool:
            if app[:-2] not in res:
                res.append(app[:-2])
        map = {}
        for cls in res:
            sub_pool = []
            for app in pool:
                if app[:-2] == cls:
                    sub_pool.append(app)
            map[cls] = sub_pool
        return res, map

    def compute_class_arrgments(self, n_class, data_set=None):
        """
        Returns all combinatorial arrangements of n items within the class present in the data.
        """
        clss, map = self.represented_classes(data_set)
        arrgments = list(itertools.combinations(clss, n_class))
        for k, x in enumerate(arrgments):
            x = list(x)
            for l, cs in enumerate(arrgments[k]):
                x[l] = np.random.choice(map[cs], 1)[0]
                arrgments[k] = x
        return arrgments

    def compute_and_save_all_dfs(self, save_dir='/home/jcgourcuff/Documents/Stage 3A/cutom_sets/', data_set=None):
        """
        Computes all new data sets and save them in save_dir.
        """
        clss, _ = self.represented_classes(data_set)
        nb_cls = len(clss)
        for k in range(2, nb_cls + 1):
            start_time = time.time()
            print("Generating {} classes data sets".format(k))
            dfs = []
            arrgments = self.compute_class_arrgments(k, data_set)
            print("{} data sets to generate\n".format(len(arrgments)))
            count = len(arrgments)
            for ar in arrgments:
                print("{} to go...".format(count), end='\r')
                dfs.append(self.make_custom_dfs_from_set_of_column_name(ar))
                count -= 1
            for l, df in enumerate(dfs):
                df.to_csv(save_dir + str(k) + '_apps/' +
                          str(l) + '.csv', compression='zip')
            print("--- %s seconds ---" % (time.time() - start_time))
        print("done")

    @staticmethod
    def get_factors():
        """
        Methods that define the number of datasets that will be generated from the data.
        """
        result = []
        for k in range(2, 9):
            nb_arrg = comb(8, k)
            i = 1
            while nb_arrg * np.power(i, k) <= 1000:
                i += 1
            i -= 1
            result.append((k, nb_arrg, i, nb_arrg * np.power(i, k)))
        result = pd.DataFrame(np.array(result), columns=[
                              'nb_class', 'nb_arrg_class', 'nb_index_per_class', 'nb_image']).set_index('nb_class')
        return result
