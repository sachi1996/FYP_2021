import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

char = cv2.imread("F:/SpellReaders/Datasets/By Class Dataset/by_class/4b-K/hsf_0/hsf_0_00024.png")
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

    horizontal_projection = np.sum(threshold, axis=1)
    return horizontal_projection


def blackPixelSum(horizontalProjection):  # calculate the sum of black pixels
    count = 0
    for i in horizontalProjection:
        count += i

    return count


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


BigZoneDensity = []

folder_images = read_image_folder(dir)

for t in range(len(folder_images)):
    full_image = folder_images[t]
    FeatureVector = []

    # break four parts from Full Image
    leftPart = full_image[0:128, 0:64]
    rightPart = full_image[0:128, 64:128]
    upperPart = full_image[0:64, 0:128]
    downPart = full_image[64:128, 0:128]

    # get Projections of each part
    left_projection = projection(leftPart)
    right_projection = projection(rightPart)
    upper_projection = projection(upperPart)
    down_projection = projection(downPart)

    # get black_pixel_sum of each part
    left = blackPixelSum(left_projection)
    right = blackPixelSum(right_projection)
    upper = blackPixelSum(upper_projection)
    down = blackPixelSum(down_projection)

    # if each sum != 0, then get 6 ratios

    total = left + right + upper + down

    if total != 0:
        percentage = (100/total)
        FeatureVector.append(left * percentage)
        FeatureVector.append(right * percentage)
        FeatureVector.append(upper * percentage)
        FeatureVector.append(down * percentage)
    """
    if left != 0 and right != 0 and upper != 0 and down != 0:
        ratio_1 = (left/right)
        ratio_2 = (upper/down)
        FeatureVector.append(ratio_1)
        FeatureVector.append(ratio_2)
    """
    BigZoneDensity.append(FeatureVector)

print(BigZoneDensity)

