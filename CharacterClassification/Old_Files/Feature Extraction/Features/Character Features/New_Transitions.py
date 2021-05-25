import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from Exract_Character import extractImage

char = cv2.imread("F:/SpellReaders/Datasets/By Class Dataset/by_class/4e-N/hsf_0/hsf_0_00024.png")
BigZoneRatio = []


def read_image_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)

    return images


def projection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_filter = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, threshold = cv2.threshold(gaussian_filter, 170, 255, cv2.THRESH_BINARY)

    threshold[gaussian_filter == 0] = 1
    threshold[gaussian_filter == 255] = 0

    return threshold


def foregroundBackround(img):

    horizontal_W_To_B = 0
    horizontal_B_To_W = 0
    vertical_W_To_B = 0
    vertical_B_To_W = 0

    img_height, img_width = img.shape

    for i in range(1, img_height):
        for j in range(1, img_width):
            if img[i][j] == 1:
                if img[i][j-1] == 1 and img[i][j+1] == 1 and img[i-1][j] == 1 and img[i+1][j] == 1:
                    break
                if img[i][j-1] == 0:
                    horizontal_W_To_B += 1
                if img[i][j+1] == 0:
                    horizontal_B_To_W += 1
                if img[i-1][j] == 0:
                    vertical_W_To_B += 1
                if img[i+1][j] == 0:
                    vertical_B_To_W += 1

    return horizontal_W_To_B, horizontal_B_To_W, vertical_W_To_B, vertical_B_To_W


####################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
dir = "G:/imwrite images/imwrite/2"

####################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

BigForeBackground = []
folder_images = read_image_folder(dir)

for t in range(len(folder_images)):
    FeatureVector = []
    img = folder_images[t]
    l, r, u, d, full = extractImage(img)

    hf, hb, vf, vb = foregroundBackround(full)

    total = hf + hb + vf + vb
    percentage = 0

    if total != 0:
        percentage = 100/total

    FeatureVector.append(hf*percentage)
    FeatureVector.append(hb*percentage)
    FeatureVector.append(vf*percentage)
    FeatureVector.append(vb*percentage)

    BigForeBackground.append(FeatureVector)


print("Done Transition")

