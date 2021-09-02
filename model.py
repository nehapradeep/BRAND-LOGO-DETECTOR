# importing necessary modules and packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import numpy as np

# getting the saved trained data from disk
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

# converting to array
y=np.array(y)

# resizing
X=X/255.0

dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            # defining the model in keras
            model = Sequential() 

            # adding the convolution layer, activation and pooling
            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            # flattening
            model.add(Flatten())

            # fully conected layer with softmax activation
            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
            model.add(Dense(4))
            model.add(Activation('softmax'))

            # creating and storing the logs
            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            # compiling the model created using appropriate loss, optimizer and evaluation metrics
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer='adam',
                          metrics=['categorical_accuracy'],)
            
            # fitting the model into the trained data
            model.fit(X, y,
                      batch_size=20,
                      epochs=40,
                      validation_split=0.1,
                      callbacks=[tensorboard])

# saving the model
model.save('CNN.model')

