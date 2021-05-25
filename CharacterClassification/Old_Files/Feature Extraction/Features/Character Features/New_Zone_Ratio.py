import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from Exract_Character import extractImage

char = "F:/Digits Dataset/White Background/Img/img002-042.png"
BigZoneRatio = []


def read_image_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)

    return images


def blackPixelSum(horizontalProjection):  # calculate the sum of black pixels
    count = 0
    for i in horizontalProjection:
        count += i

    return count


BigZoneDensity = []

####################################################
# A ---> dir = "F:/Character Dataset/A/Sharp A"
# dir = "F:/Character Dataset/B/Sharp B"
# dir = "F:/Character Dataset/bb/Sharp b"
# dir = "F:/Character Dataset/C/Sharp C"
# dir = "F:/Character Dataset/d/Sharp d"
# dir = "F:/Character Dataset/e/Sharp e"
# dir = "F:/Character Dataset/F/Sharp F"
# dir = "F:/Character Dataset/ff/Sharp f"
# dir = "F:/Character Dataset/g/Sharp g"
# dir = "F:/Character Dataset/H/Sharp H"
# dir = "F:/Character Dataset/hh/Sharp h"
# dir = "F:/Character Dataset/I Capital--/Sharp I"
# dir = "F:/Character Dataset/K/Sharp K"
# dir = "F:/Character Dataset/l simple/I1"
# dir = "F:/Character Dataset/M/Sharp M"
# dir = "F:/Character Dataset/N/Sharp N"
# dir = "F:/Character Dataset/nn/Sharp n"
# dir = "F:/Character Dataset/O/Sharp O"

####################################################

####################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
dir = "G:/imwrite images/imwrite/2"

####################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

folder_images = read_image_folder(dir)
whole_images = []

for t in range(len(folder_images)):
    full_image = folder_images[t]
    FeatureVector = []

    left, right, upper, down, fully = extractImage(full_image)

    left_proj = np.sum(left, axis=1)
    right_proj = np.sum(right, axis=1)
    upper_proj = np.sum(upper, axis=1)
    down_proj = np.sum(down, axis=1)

    left_sum = blackPixelSum(left_proj)
    right_sum = blackPixelSum(right_proj)
    upper_sum = blackPixelSum(upper_proj)
    down_sum = blackPixelSum(down_proj)

    total = left_sum + right_sum + upper_sum + down_sum

    percentage = (100/total)

    FeatureVector.append(left_sum * percentage)
    FeatureVector.append(right_sum * percentage)
    FeatureVector.append(upper_sum * percentage)
    FeatureVector.append(down_sum * percentage)

    BigZoneDensity.append(FeatureVector)

print("Done Zone")

