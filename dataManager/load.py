import numpy as np
import pandas as pd


class DataSets():
    """
        Class to embed extractors and basic preprocessing for each data set that will be used in this project.
        Datasets are returned in the form of a Pandas DataFrame, each columns corresponds to one appliance and
        index is the timestamp of each sample given as an integer representing seconds.
    """

    def __init__(self):
        """
        This class aims to hold static functions only.
        """
        return None

    @staticmethod
    def REDD(house_nb, demo=False):
        """
            :param house_nb: Integer from 1 to 6
            :return: A dataframe containing all data for individual appliances monitored in the house. Index is timestamp.
        """

        file_loc = "data/REDD/low_freq/house_" + str(house_nb) + "/"

        if demo:
            assert house_nb == 1
            file_loc = file_loc = "data/REDD_demo/low_freq/house_" + str(house_nb) + "/"

        labels = pd.read_csv(file_loc + "labels.dat",
                             sep=" ", names=["id", "name"])

        dfs = []
        for id, name in zip(labels['id'], labels['name']):
            if 'mains' not in name:
                loc = file_loc + "channel_" + str(id) + ".dat"
                dfs.append(pd.read_csv(loc, sep=" ", names=[
                    "timestamp", name]).set_index('timestamp'))

        appliances = pd.concat(dfs, axis=1)

        return appliances

    @staticmethod
    def BCR(phase_nb):
        """
            To load data from BCR's house. Remark: data has multiple values per minute, but not second information...
            :param phase_nb: Integer from 1 to 3, phase number.
            :return: A dataframe containing the aggregated phase load, with a datetime as index Paris time zone info.
        """
        file_loc = "data/BCR/Load_L" + str(phase_nb) + ".zip"  # I store zipped csvs to save space.
        mydateparser = lambda x: pd.datetime.strptime(x, "%d/%m/%Y %H:%M")
        df = pd.read_csv(file_loc, sep=",", names=["dateTime", "Load"], parse_dates=['dateTime'],
                         date_parser=mydateparser, index_col="dateTime")
        df.index = df.index.tz_localize('UTC').tz_convert('Europe/Paris')
        return df
