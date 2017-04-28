import tensorflow as tf
import random
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("skive/normal_img/0.png").convert("RGB")
plt.imshow(img)
plt.show()

pixels = img.load()
(width, height) = img.size

for y in range(height):
    for x in range(width):
        print(y, x, pixels[x,y])
