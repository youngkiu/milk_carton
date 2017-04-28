import os
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


def loadImage(imageFile, labelFile):
    xImage = Image.open(imageFile).convert("L")
    yLabel = Image.open(labelFile).convert("L")
    plt.imshow(xImage, cmap='Greys')
    plt.show()
    plt.imshow(yLabel, cmap='Greys')
    plt.show()

    xPix = np.array(xImage)
    yPix = np.array(yLabel)
    print(xPix.shape)
    print(yPix.shape)

    # data image of shape 300 * 1660 * 1 = 498000
    #(width, height) = xImage.size
    #for y in range(height):
    #    for x in range(width):
    #        print(y, x, xPix[x,y], yPix[x,y])

    x = np.concatenate(xPix)
    y = np.concatenate(yPix)
    xy = np.vstack((x, y))
    print(xy.shape)

    return xy


if __name__ == '__main__':
    #loadImage("skive/normal_img/0.png")
    xy = loadImage("skive/defect_img/image_0.png", "skive/defect_img/label_0.png")
