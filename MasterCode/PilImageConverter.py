import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#import cv2
import os

from PIL import Image, ImageEnhance
from pathlib import Path
import matplotlib.pyplot as plt
import uuid
import random
import os


class ImagePreProcessor():
    def __init__(self):
        pass

    def sharpen(img, factor):
        enhancer_sharpness = ImageEnhance.Sharpness(img)
        return enhancer_sharpness.enhance(factor)

    def contrast(img, factor):
        enhancer_contrast = ImageEnhance.Contrast(img)
        return enhancer_contrast.enhance(factor)

    def color(img, factor):
        enhancer_color = ImageEnhance.Color(img)
        return enhancer_color.enhance(factor)

    def rotate(img, degrees):
        return img.rotate(degrees)

    def save(img, path):
        return img.save(path, "PNG")

    def plotImages(pathList, number):
        pp = list(pathList)
        plt.figure(figsize=(15, 15))
        for i in range(number):
            plt.subplot(5, 5, i + 1)
            im = Image.open(pp[i])
            plt.imshow(im)
            plt.xticks([])
            plt.yticks([])
        plt.show()

PreprocessorImg = ImagePreProcessor



inputBasePath = "/Users/manishreddybendhi/PycharmProjects/CupsAndBottlesClassifier/Bottles_Cups_Tumblers"
outputBasePath = "/Users/manishreddybendhi/PycharmProjects/CupsAndBottlesClassifier/Bottles_Cups_Tumblers_Output"




folders = ["BOTTLES","COFFEEMUGS","CUPS"]


def PreprocessingImgs(inputBasePath,outputBasePath,folders):
    left = 500
    top = 500
    right = 500
    bottom = 500
    rotations = [0, 90, 180, 270]
    randContrastMin, randContrastMax = (0.8, 1.2)
    randSharpenMin, randSharpenMax = (0.8, 1.2)
    randColorMin, randColorMax = (0.8, 1.2)
    multiplier = 2

    for f in folders:
        # print(inputBasePath+f+"/")
        plist = Path(inputBasePath + '/' + f + '/').glob('*.jpg')

        outpath = outputBasePath + '/' + f + '/'
        print(outpath)
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        for path in plist:
            i = Image.open(path)
            # crop_img = i.crop((left, top, right, bottom))
            i = i.resize((224, 224), Image.ANTIALIAS)

            for r in rotations:

                for m in range(multiplier):
                    randContrast = random.uniform(randContrastMin, randContrastMax)
                    randSharpen = random.uniform(randSharpenMin, randSharpenMax)
                    randColor = random.uniform(randColorMin, randColorMax)

                    i = ImagePreProcessor.rotate(i, r)
                    i = ImagePreProcessor.contrast(i, randContrast)
                    i = ImagePreProcessor.sharpen(i, randSharpen)
                    i = ImagePreProcessor.color(i, randColor)

                    ImagePreProcessor.save(i, outpath + str(uuid.uuid4()) + '.jpg')
                    print(outpath)
                    print('.', end='')
PreprocessingImgs(inputBasePath,outputBasePath,folders)