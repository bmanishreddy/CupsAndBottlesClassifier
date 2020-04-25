from pathlib import Path
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import vgg16
import keras
import re

#BOTTLES = Path('/Users/manishreddybendhi/PycharmProjects/SimpleImageClassifier/Bottles_Cups_Tumblers_Output/BOTTLES/')
#COFFEEMUGS = Path('/Users/manishreddybendhi/PycharmProjects/SimpleImageClassifier/Bottles_Cups_Tumblers_Output/COFFEEMUGS/')
#CUPS = Path('/Users/manishreddybendhi/PycharmProjects/SimpleImageClassifier/Bottles_Cups_Tumblers_Output/CUPS/')
PathFileDir = "/Users/manishreddybendhi/PycharmProjects/CupsAndBottlesClassifier/Bottles_Cups_Tumblers_Output/"

def PreProcessesImgs(PathFileDir):
    BOTTLES = Path(PathFileDir + 'BOTTLES/')
    COFFEEMUGS = Path(PathFileDir + 'COFFEEMUGS/')
    CUPS = Path(PathFileDir + 'CUPS/')
    images = []
    labels = []
    ObjectImgs = [BOTTLES, CUPS, COFFEEMUGS]
    for j in ObjectImgs:
        for img in j.glob("*.jpg"):
            img = image.load_img(img)
            image_array = image.img_to_array(img)
            if str(j) == PathFileDir + "BOTTLES":
                images.append(image_array)
                labels.append([0])
            elif str(j) == PathFileDir + "CUPS":
                images.append(image_array)
                labels.append([1])
            elif str(j) == PathFileDir + "COFFEEMUGS":
                images.append(image_array)
                labels.append([2])
    x_train = np.array(images)
    y_train = np.array(labels)
    return x_train,y_train


def SaveDat():
    x_train,y_train = PreProcessesImgs(PathFileDir)
    y_train = keras.utils.to_categorical(y_train, 3)
    print(y_train)
    x_train = vgg16.preprocess_input(x_train)
    pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    features_x = pretrained_nn.predict(x_train)
    joblib.dump(features_x, "x_train.dat")
    joblib.dump(y_train, "y_train.dat")


SaveDat()