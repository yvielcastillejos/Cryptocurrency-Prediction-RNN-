import pandas as pd
import csv
import numpy as np
import os
from funct import*
import tensorflow as tf
from sklearn import preprocessing
from collections import deque
import random
from RNN import*

category_to_predict = "ETH-USD"
Future_to_predict = 3 #min whether to buy it after 3 minutes


def main():
    DIR = "/Users/yvielcastillejos/Downloads/crypto_data"
    CATEGORIES = os.listdir(DIR)  # ['ETH-USD', 'BTC-USD', 'BCH-USD', 'LTC-USD']
    validation_set = pd.DataFrame()
    # Define Param
    SEQ_DATA = 60  # min
    Future_to_predict = 3  # 3 min after
    category_to_predict = "LTC-USD"
    # UNIX labels (columns)
    classes = ["timestamp", "Low", "High", "Open", "Close", "Volume"]
    # define frames
    data_frame = pd.DataFrame()
    # Fix name
    for number in range(0, len(CATEGORIES), 1):
        CATEGORIES[number] = CATEGORIES[number][:-4]
    for category in CATEGORIES:
        path = os.path.join(DIR, f"{category}.csv")
        dataframe = pd.read_csv(path, names=classes)
        dataframe.rename(columns={"Close": f"{category}_Close", "Volume": f"{category}_Volume"}, inplace=True)
        dataframe.set_index("timestamp", inplace=True)
        dataframe = dataframe[[f"{category}_Close", f"{category}_Volume" ]]
        #print(dataframe.head())
        data_frame = merge_1(dataframe, data_frame)
    data_frame.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
    data_frame.dropna(inplace=True)
    data_frame['future'] = data_frame[f"{category_to_predict}_Close"].shift(-Future_to_predict)
    data_frame["to_buy"] = list(map(classify, data_frame[f"{category_to_predict}_Close"], data_frame['future']))
    #print(data_frame[[f"{category_to_predict}_Close", "future", "to_buy"]])
    validation_set, data_frame = Get_ValidationData(data_frame)
    # print(data_frame.head())
    train_x, train_y = Normalize(data_frame)
    validation_x, validation_y = Normalize(validation_set)
    np.save("train_x.npy",train_x)
    np.save("train_y.npy",train_y)
    np.save("validation_x.npy",validation_x)
    np.save("validation_y.npy",validation_y)
    return

main()
train()
