import pandas as pd
import numpy as np
from sklearn import preprocessing

from keras.models import load_model

import os

ruta = os.getcwd() + '/data'
ruta_train = ruta + '/train.csv'
ruta_test = ruta + '/test.csv'

df = pd.read_csv(ruta_train)
df = df[df.columns[2:]]

df_test = pd.read_csv(ruta_test)
df_test = df_test[df_test.columns[1:]]


def transf_feat(features):
    _vect = features[0]
    dim = int(len(_vect)**0.5) + 1
    zeros  = dim**2 -len(_vect)
    z = [0]*(zeros - dim)

    feat_imags = np.array([np.concatenate([row, z]).reshape((-1, dim)) for row in features])

    x = feat_imags.reshape(feat_imags.shape[0], feat_imags.shape[1], feat_imags.shape[2], 1)

    return x

x_test = transf_feat(feat_test)

model = load_model('Model_Santander_GPU_v1.1.60.h5')


def Predict(ima_test, model=model):
    y_pred = model.predict(ima_test.reshape((1,14,15,1)))
    prediccion = list(y_pred[0]).index(y_pred[0].max())
    return prediccion


pred_dict = {}
size = float(len(x_test))
for indice, imagen in enumerate(x_test):
    pred_dict['test_' + str(indice)] = Predict(imagen)
    print(str(round((float(indice)/size)*100, 2)) + '%')

df1 = pd.DataFrame({'ID_code': list(pred_dict.keys()), 'target': list(pred_dict.values())}).set_index('ID_code')

df1.to_csv('./Output/Santander2_out_v1.1.60.csv')
