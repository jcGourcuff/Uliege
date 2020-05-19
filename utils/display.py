import matplotlib.pyplot as plt

FIG_DIM = (20, 8)


class Graphs:
    """
    Class to embed graph functions.
    """

    def __init__(self):
        """
        This class is aimed to contain static methods only.
        """
        return None

    @staticmethod
    def aggregate(data_set, unit='watt', title="Aggregated power"):
        """
        Draws the aggregated power of all appliances in the data frame, given
        that each column correspond to an appliance's consumption.

        :param data_set: Pandas DataFrame that holds appliances consumption.
        :param unit: Unit of Power to appear on the graph. Default 'watt'.
        :param title: Title of the figure. Default "Aggregated power".
        """
        aggregated = data_set.sum(axis=1)
        fig = plt.figure(figsize=FIG_DIM)
        plt.plot(aggregated, color='black')
        fig.autofmt_xdate()
        plt.ylabel(unit)
        plt.title(title)
        plt.show()

    @staticmethod
    def decompose_aggregate(data_set, unit='watt', title="Appliances contribution to the aggregated power"):
        """
        Draws the aggregate power of all appliances in the dataframe with a
        comprehensive color filling, given that each column correspond to an
        appliance's consumption.

        :param data_set: Pandas DataFrame that holds appliances consumption.
        :param unit: Unit of Power to appear on the graph. Default 'watt'.
        :param title: Title of the figure. Default "Appliances contribution to the aggregated power".
        """
        aggregated = data_set.sum(axis=1)
        fig = plt.figure(figsize=FIG_DIM)
        for name in data_set.columns:
            next = aggregated - data_set[name]
            plt.fill_between(aggregated.index, next, aggregated, label=name)
            aggregated = next
        plt.ylabel(unit)
        plt.title(title)
        fig.autofmt_xdate()
        plt.legend()
        plt.show()

    @staticmethod
    def draw(serie, unit='watt', title=''):
        """
        Draws the time serie given as an argument.

        :param serie: Pandas Series that holds the data.
        :param unit: Unit of Power to appear on the graph. Default 'watt'.
        :param title: Title of the figure. Default None.
        """
        fig = plt.figure(figsize=FIG_DIM)
        plt.plot(serie)
        plt.title(title)
        plt.ylabel(unit)
        fig.autofmt_xdate()
        plt.show()
