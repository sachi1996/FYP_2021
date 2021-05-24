import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

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

    for i in range(1, 126):
        for j in range(1, 126):
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


# dir = "F:/Character Dataset/A/A9"
# dir = "F:/Character Dataset/b/b4"
# dir = "F:/Character Dataset/C/c4"
# dir = "F:/Character Dataset/d/d2"
# dir = "F:/Character Dataset/e/e1"
# dir = "F:/Character Dataset/F/f4"
# dir = "F:/Character Dataset/ff/f7"
# dir = "F:/Character Dataset/I Capital/I1"
# dir = "F:/Character Dataset/i simple/i6"
# dir = "F:/Character Dataset/K/K7"
# dir = "F:/Character Dataset/m/m7"
# dir = "F:/Character Dataset/n/n2"
# dir = "F:/Character Dataset/O/O1"
dir = "F:/Character Dataset/T/T1"


BigForeBackground = []
folder_images = read_image_folder(dir)

for t in range(len(folder_images)):
    FeatureVector = []
    full_image = folder_images[t]
    projection_img = projection(full_image)
    hf, hb, vf, vb = foregroundBackround(projection_img)

    total = hf + hb + vf + vb
    percentage = 0

    if total != 0:
        percentage = 100/total

    FeatureVector.append(hf*percentage)
    FeatureVector.append(hb*percentage)
    FeatureVector.append(vf*percentage)
    FeatureVector.append(vb*percentage)

    BigForeBackground.append(FeatureVector)


print(BigForeBackground)

