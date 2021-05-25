import cv2
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os

character_z = "F:/SpellReaders/Datasets/By Class Dataset/by_class/Z/hsf_1/hsf_1_00006.png"
character_k = "E:/SpellReaders/Datasets/By Class Dataset/by_class/6b/hsf_1/hsf_1_00005.png"


image_z = cv2.imread(character_z)
name = plt.imshow(image_z)
plt.title("Character - Z")
plt.show()

Img_Original = cv2.imread(character_z)
FeatureVector = []


def drawZoneLines(Img_Original):
    h = cv2.line(Img_Original, (64, 0), (64, 128), (102, 102, 255), 1)
    cv2.line(h, (0, 64), (128, 64), (102, 255, 102), 1)
    return Img_Original


AfterDrawLines = drawZoneLines(image_z)
plt.imshow(AfterDrawLines)
plt.show()
