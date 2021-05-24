import matplotlib.pyplot as plt
import cv2
import os
from Exract_Character import extractImage


def read_image_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)

    return images


def chaincode(img):
    direction1 = 0
    direction2 = 0
    direction3 = 0
    direction4 = 0
    direction5 = 0
    direction6 = 0
    direction7 = 0
    direction8 = 0

    img_height, img_width = img.shape

    for i in range(1, img_height):
        for j in range(1, img_width):
            if img[i][j] == 1:
                if (img[i-1][j] == 1 and img[i-1][j+1] == 1 and img[i][j+1] == 1 and img[i+1][j+1] == 1
                        and img[i+1][j] == 1 and img[i+1][j-1] == 1 and img[i][j - 1] == 1 and img[i-1][j-1] == 1):
                    break
                else:
                    if img[i-1][j] == 1:
                        direction1 += 1
                    if img[i-1][j+1] == 1:
                        direction2 += 1
                    if img[i][j+1] == 1:
                        direction3 += 1
                    if img[i+1][j+1] == 1:
                        direction4 += 1
                    if img[i+1][j] == 1:
                        direction5 += 1
                    if img[i+1][j-1] == 1:
                        direction6 += 1
                    if img[i][j-1] == 1:
                        direction7 += 1
                    if img[i-1][j-1] == 1:
                        direction8 += 1

    return direction1, direction2, direction3, direction4, direction5, direction6, direction7, direction8


####################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
dir = "G:/imwrite images/imwrite/2"

####################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


images = read_image_folder(dir)

BigChainCode = []
for k in range(len(images)):
    img = images[k]
    l, r, u, d, full = extractImage(img)

    Vector = []

    d1, d2, d3, d4, d5, d6, d7, d8 = chaincode(full)
    total = d1 + d2 + d3 + d4 + d5 + + d6 + d7 + d8

    percentage = 100 / total

    Vector.append(d1 * percentage)
    Vector.append(d2 * percentage)
    Vector.append(d3 * percentage)
    Vector.append(d4 * percentage)
    Vector.append(d5 * percentage)
    Vector.append(d6 * percentage)
    Vector.append(d7 * percentage)
    Vector.append(d8 * percentage)

    BigChainCode.append(Vector)


print("Done ChainCode")
# [775, 743, 750, 724, 775, 743, 750, 724]
