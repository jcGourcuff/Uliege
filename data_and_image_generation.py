import pandas as pd
import numpy as np
from dataManager.laod import DataSets
import time
from nilm.imageGen_and_labelling import LabeledImagesMaker, ArtificialSetGenerator

"""
This script shows how the datasets for the computer vision framework were created.
However, it is highly not recommended to run this script since it is computationnaly ultra heavy.
"""


#Import data set
df = DataSets.REDD(1)

#resample and drop unwanted columns
sampled = df.resample('1min').fillna('ffill')[1:]
columns_drops = {'1': ['oven_1', 'kitchen_outlets_1', 'washer_dryer_2', 'electric_heat_1', 'stove_1']}
sampled = sampled[sampled.columns.drop(columns_drops['1'])]

#Instanciate Data set generator
setGen = ArtificialSetGenerator(sampled)

#Visualize how many days of data will be generated
setGen.get_factors()

#Generate and save the data sets
setGen.compute_and_save_all_dfs(data_set=sampled, save_dir='/home/jcgourcuff/Documents/Stage 3A/cutom_sets/')

#Set paths to csvs
dir = '/home/jcgourcuff/Documents/Stage 3A/image_custom_sets/'

#This next method build image data set and labels csvs
def make_image_datasets():
    start_id = 1
    for k in range(2,9):
        start_time = time.time()
        print("n apps {}".format(k))
        for l in range(int(setGen.get_factors().loc[k,'nb_arrg_class'])):
            print("data set n {}".format(l))
            a = pd.read_csv(dir + str(k) +'_apps/' +  str(l) + '.csv', compression = 'zip')
            a.columns = ['timestamp']+list(a.columns[1:])
            a['timestamp'] = pd.to_datetime(a['timestamp'])
            a = a.set_index('timestamp')

            directory = dir + str(k) +'_apps/'
            start_id = LabeledImagesMaker.make_data_set(data = a, directory=directory, label_num=l, start_id = start_id)

            del a
            print("--- %s seconds ---" % (time.time() - start_time))

#This next methods compile all labels files into one for each number of aplliance within images
def compile_label_sets():
    for k in range(2,9):
        dfs = []
        for l in range(int(setGen.get_factors().loc[k,'nb_arrg_class'])):
            df = pd.read_csv(dir + str(k) +'_apps/labels_'  + str(l) + '.csv')
            if df['label'].drop_duplicates().shape[0] != k + 1 :
                print("Warning {} classes instead of {} ; k = {}, l = {}".format(df['label'].drop_duplicates().shape[0]-1, k, k, l))
                print(df['label'].drop_duplicates())


            dfs.append(df)
        dfs = pd.concat(dfs, axis = 0, ignore_index = True)
        dfs.to_csv(dir+ str(k) +'_apps/labels.csv', index = False, compression = 'zip')
