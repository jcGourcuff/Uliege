import numpy as np
import torch
import torch.nn as nn

class data_manager():
    """
    Class to implement a buffer. Methods are self explanatory.
    """

    def __init__(self, df_size):
        self.df = []
        self.df_size_ = df_size

    def reset(self):
        self.df = []

    def is_full(self):
        return len(self.df) == self.df_size_

    def add(self, line):
        if self.is_full():
            print("already full")
        else:
            self.df.append(line)

    def get_df(self):
        return np.array(self.df)

    def get_size(self):
        return len(self.df)

    def get_random_batch(self, batch_size):
        idxs = np.random.randint(len(self.df), size = batch_size)
        batch = self.get_df()[idxs,:]
        return batch

    def dump(self, x):
        nb = int((1 - x) * self.get_size())
        idx = np.random.randint(
            low=0, high=self.get_size(), size=nb, dtype=int)
        self.df = list(np.array(self.df)[idx])

    def update_data(self, option = 'add', index = None):
        """
        Method that allows to update the buffer by removing relevant part of the data so the buffer doesn't have to be emptied
        when used for a dynamic output neural network.
        """
        new_df = []
        if option == 'add' :
            for x in self.df :
                if type(x)!= list :
                    x = x.tolist()
                new_df.append(x[:-1] + [x[-1] + [0]])
            del self.df
            self.df = new_df

        elif option == 'remove' :
            assert index is not None
            for x in self.df :
                if type(x)!= list :
                    x = x.tolist()
                new_df.append(x[:-1] + [x[-1][:index]+x[-1][index+1:]])
            del self.df
            self.df = new_df


class AppNet(nn.Module):
    """
    Implements the NN that outputs the proba at time t that the corresponding app is on.
    """

    def __init__(self, input_size):
        super(AppNet, self).__init__()
        self.input_size = input_size

        self.common_layers = self.layer_0 = nn.Sequential(nn.Linear(self.input_size,self.input_size),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm1d(self.input_size),
                                    nn.Dropout(inplace = False),
                                    nn.Linear(self.input_size,self.input_size),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm1d(self.input_size),
                                    nn.Dropout(inplace = False),
                                    nn.Linear(self.input_size,self.input_size),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm1d(self.input_size),
                                    nn.Dropout(inplace = False),
                                    nn.Linear(self.input_size,self.input_size),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm1d(self.input_size))

        self.fc = nn.Sequential(nn.Dropout(inplace=False),
                                nn.Linear(self.input_size, self.input_size),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(self.input_size),
                                nn.Dropout(inplace = False),
                                nn.Linear(self.input_size,1))

    def forward(self, x):
        x = self.common_layers(x)
        x = self.fc(x)
        return x
