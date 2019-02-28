import pandas as pd
import numpy as np
from sklearn import preprocessing

from keras.models import load_model

import os

ruta = os.getcwd() + '/data'
ruta_train = ruta + '/train.csv'
ruta_test = ruta + '/test.csv'

df = pd.read_csv(ruta_train)
df1 = df[df.columns[2:]]

sc = preprocessing.MinMaxScaler(feature_range=(0, 1))
features = sc.fit_transform(df1)
target = df.target.values

def transf_feat(features):
    _vect = features[0]
    dim = int(len(_vect)**0.5) + 1
    zeros  = dim**2 -len(_vect)
    z = [0]*(zeros - dim)

    feat_imags = np.array([np.concatenate([row, z]).reshape((-1, dim)) for row in features])

    x = feat_imags.reshape(feat_imags.shape[0], feat_imags.shape[1], feat_imags.shape[2], 1)

    return x

x_train = transf_feat(features)
y_train = np_utils.to_categorical(target,2)

model = load_model('Model_newNN_GPU_v5.11.30.h5')


