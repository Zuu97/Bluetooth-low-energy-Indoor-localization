import numpy as np
import pandas as pd
from variables import*
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def mlp_data():
    df = pd.read_csv(data_path, index_col=None)
    del df['date']

    df = shuffle(df)
    cols = df.columns.values

    locations = df[cols[0]].values
    encoder = LabelEncoder()
    locations = encoder.fit_transform(locations)

    rssi = df[cols[1:]].values
    scalar = MinMaxScaler()
    rssi = scalar.fit_transform(rssi)

    return locations, rssi

def ann_data():
    df = pd.read_csv(data_path)
    df['x'] = df['location'].str[0]
    df['y'] = df['location'].str[1:]
    df.drop(["location"], axis = 1, inplace = True)
    df["x"] = df["x"].apply(fix_pos)
    df["y"] = df["y"].astype(int)

    colmns = df.columns.values
    Y = df[colmns[-2:]].values
    X = df[colmns[1:-2]].values

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
                                            X,Y,
                                            test_size = validation_split,
                                            shuffle = False
                                            )
    return Xtrain, Ytrain, Xtest, Ytest

def eucledian_distance(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    x1, y1 = np.array(x1), np.array(y1)
    x2, y2 = np.array(x2), np.array(y2)
    dx = x1 - x2
    dy = y1 - y2
    dists = np.sqrt(dx ** 2 + dy ** 2)
    return np.mean(dists), dists

def fix_pos(x_cord):
    x = 87 - ord(x_cord.upper())
    return x

def rf_data():
    locations, rssi = mlp_data()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
                                            rssi,locations,
                                            test_size = validation_split,
                                            shuffle=True
                                            )
    return Xtrain, Ytrain, Xtest, Ytest