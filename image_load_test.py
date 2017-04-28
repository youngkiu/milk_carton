import os
import matplotlib.pyplot as plt
import numpy as np
import random

from PIL import Image


def loadImage(imageFile, labelFile):
    print(imageFile, labelFile)

    xImage = Image.open(imageFile).convert("L")
    yLabel = Image.open(labelFile).convert("L")
    #plt.imshow(xImage, cmap='Greys')
    #plt.show()
    #plt.imshow(yLabel, cmap='Greys')
    #plt.show()

    xPix = np.array(xImage)
    yPix = np.array(yLabel)
    #print(xPix.shape)
    #print(yPix.shape)

    # data image of shape 300 * 1660 * 1 = 498000
    #(width, height) = xImage.size
    #for y in range(height):
    #    for x in range(width):
    #        print(y, x, xPix[x,y], yPix[x,y])

    x = np.concatenate(xPix)
    y = np.concatenate(yPix)
    xy = np.vstack((x, y))
    #print(xy.shape)

    return xy

def selectTrainingData():
    normal_img_dir = "skive/normal_img/"
    defect_img_dir = "skive/defect_img/"

    numOfNormalImg = len(os.listdir(normal_img_dir)) - 1 # All image share a label
    numOfDefectImg = len(os.listdir(defect_img_dir)) / 2 # Each image has a label
    numOfTrainingData = numOfNormalImg + numOfDefectImg

    selectedIndex = random.randint(0, numOfTrainingData - 1)
    #print(selectedIndex, numOfNormalImg, numOfDefectImg, numOfTrainingData)

    if selectedIndex < numOfNormalImg:
        fileIndex = 0
        files = os.listdir(normal_img_dir)
        for file in files:
            if file.find("label") < 0:
                if fileIndex == selectedIndex:
                    xy = loadImage("{0}{1}".format(normal_img_dir, file), \
                                   "{0}{1}".format(normal_img_dir, "label.png"))
                    break
                fileIndex += 1
    else:
        fileIndex = numOfNormalImg
        files = os.listdir(defect_img_dir)
        for file in files:
            if file.find("image") >= 0:
                if fileIndex == selectedIndex:
                    xy = loadImage("{0}{1}".format(defect_img_dir, file), \
                                   "{0}{1}".format(defect_img_dir, file.replace("image", "label")))
                    break
                fileIndex += 1

    return xy

if __name__ == '__main__':
    random.seed(20170428)

    for i in range(1000):
        xy = selectTrainingData()
