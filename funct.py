import pandas as pd
import csv
import numpy as np
import os
import funct
import tensorflow as tf
from sklearn import preprocessing
from collections import deque
import random

# Merge df
def merge_1(df, main_dataframe):
    if len(main_dataframe) == 0:
        main_dataframe = df
    else:
        main_dataframe = main_dataframe.join(df)
    return main_dataframe

# Classify
def classify(current, future):
    if float(current) > float(future):
        return 0
    else:
        return 1

# Validation data
def Get_ValidationData(df):
    # Gets the last 5%
    times = sorted(df.index.values)
    validation_set = df[(df.index >= times[int(-(0.05*len(times)))])]
    df = df[df.index< times[int(-(0.05*len(times)))]]
    return validation_set, df

# Normalize
def Normalize(df):
    df = df.drop("future", 1)
    for col in df.columns:  # go through all of the columns
        if col != "to_buy":  # normalize all ... except for the target itself!
            df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.dropna(inplace=True)  # remove the nas created by pct_change
            #print(df.head)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)
    sequential_data = []
    prev_days = deque(maxlen=60)
    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == 60:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])
        random.shuffle(sequential_data)
    buys = []
    sells = []
    X = []
    Y= []
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq,target])
        else:
            buys.append([seq,target])
    random.shuffle(sells)
    random.shuffle(buys)
    lower = min(len(buys),len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    sequential_data = buys+sells
    random.shuffle(sequential_data)
    for seq, target in sequential_data:
        X.append(seq)
        Y.append(target)
    return np.array(X), Y
