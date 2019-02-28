import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model
from keras.layers.advanced_activations import PReLU,LeakyReLU

ruta = os.getcwd() + '/data'
ruta_train = ruta + '/train.csv'

df = pd.read_csv(ruta_train)
target = df.target.values
df = df[df.columns[2:]]

features= df.values

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


"""
Arquitectura Red Neuronal
"""

model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='linear', input_shape = (x_train.shape[1],x_train.shape[2],1)))
model.add(PReLU())
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='linear'))
model.add(PReLU())
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='linear'))
model.add(PReLU())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))


model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                 activation ='linear'))
model.add(PReLU())
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                 activation ='linear'))
model.add(PReLU())
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='linear'))
model.add(PReLU())
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='linear'))
model.add(PReLU())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation = "sigmoid"))


optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])


# funcion para modificar el factor de aprendizaje en funcion de su evolucion
learning_rate_reduction = ReduceLROnPlateau(monitor='acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1)
datagen.fit(x_train)


epochs = 60 #
batch_size = 86
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs,
                              verbose = 1,
                              # steps_per_epoch = x_train.shape[0] // batch_size,
                              steps_per_epoch = 1000,
                              callbacks = [learning_rate_reduction])

# Persistimos el modelo
model.save('Model_Santander_GPU_v1.0.60.h5')