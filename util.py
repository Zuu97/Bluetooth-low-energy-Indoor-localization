import numpy as np
import pandas as pd
from variables import*
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def get_data():
    df = pd.read_csv(data_path)
    del df['date']

    df = shuffle(df)
    cols = df.columns.values

    locations = df[cols[0]].values
    encoder = LabelEncoder()
    locations = encoder.fit_transform(locations)

    rssi = df[cols[1:]].values
    scalar = StandardScaler()
    rssi = scalar.fit_transform(rssi)

    return locations, rssi
