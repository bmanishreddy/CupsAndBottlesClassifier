from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import keras
import joblib
import tensorflow as tf




def Train_save(X_train,Y_train):
    # Load data set
    #x_train = joblib.load("x_train.dat")
    #y_train = joblib.load("y_train.dat")
    x_train = joblib.load(X_train)
    y_train = joblib.load(Y_train)
    # y_train = np.split(y_train,240)
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    # Compile the model
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    # print(model.summary())
    print(y_train)
    # Train the model
    model.fit(x_train,y_train,epochs=20,shuffle=True)
    # Save neural network structure
    model_structure = model.to_json()
    f = Path("model_structure.json")
    f.write_text(model_structure)

    # Save neural network's trained weights
    model.save_weights("model_weights.h5")

Train_save("x_train.dat","y_train.dat")