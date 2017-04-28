import os
import numpy as np

from PIL import Image


def load_train_data():
    normal_img_dir = "skive/normal_img/"
    defect_img_dir = "skive/defect_img/"

    x_data = []
    y_data = []

    files = os.listdir(normal_img_dir)
    for file in files:
        filename = "{0}{1}".format(normal_img_dir, file)
        img = Image.open(filename).convert("L")
        pix = np.array(img)

        if x_data == []:
            x_data = [np.concatenate(pix)]
            y_data = [[1, 0]]
        else:
            x_data = np.concatenate((x_data, [np.concatenate(pix)]), axis=0)
            y_data = np.concatenate((y_data, [[1, 0]]), axis=0)

    files = os.listdir(defect_img_dir)
    for file in files:
        if file.find("image") >= 0:
            filename = "{0}{1}".format(defect_img_dir, file)
            img = Image.open(filename).convert("L")
            pix = np.array(img)

            x_data = np.concatenate((x_data, [np.concatenate(pix)]), axis=0)
            y_data = np.concatenate((y_data, [[0, 1]]), axis=0)

    return x_data, y_data

if __name__ == '__main__':
    x_data, y_data = load_train_data()
    print(x_data.shape)
    print(y_data.shape)

    num_of_data = np.size(x_data, 0)
    image_size = np.size(x_data, 1)
    nb_classes = np.size(y_data, 1)

