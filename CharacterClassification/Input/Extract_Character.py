from builtins import str

import cv2
import matplotlib.pyplot as plt
import numpy as np

def thresholding(image):
    camScanner = cv2.imread(image)
    grayCamScanner = cv2.cvtColor(camScanner, cv2.COLOR_BGR2GRAY)
    GaussianFiltered5 = cv2.GaussianBlur(grayCamScanner, (5, 5), 0)
    _, GaussianThresh5 = cv2.threshold(GaussianFiltered5, 127, 255, cv2.THRESH_BINARY)

    original = GaussianThresh5.copy()

    height, width = original.shape

    original[original == 0] = 1
    original[original == 255] = 0

    return original


def extractImage(inputImage):
    grayCamScanner = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    GaussianFiltered5 = cv2.GaussianBlur(grayCamScanner, (5, 5), 0)
    _, GaussianThresh5 = cv2.threshold(GaussianFiltered5, 127, 255, cv2.THRESH_BINARY)

    original = GaussianThresh5.copy()

    height, width = original.shape

    original[original == 0] = 1
    original[original == 255] = 0

    horizontal_projection = np.sum(original, 1)
    vertical_projection = np.sum(original, 0)

    row_index = 0
    column_index = 0
    horizontal_start = 0
    horizontal_end = 0
    vertical_start = 0
    vertical_end = 0

    for pixel_count in horizontal_projection:
        # print(row_index + 1, pixel_count)
        if pixel_count == 0:
            if row_index < (height - 1):
                if (horizontal_projection[row_index - 1] == 0) and (horizontal_projection[row_index + 1] != 0):
                    vertical_start = row_index
                if (horizontal_projection[row_index - 1] != 0) and (horizontal_projection[row_index + 1] == 0):
                    vertical_end = row_index + 2
                else:
                    pass
            else:
                pass
        else:
            pass

        row_index = row_index + 1

    for pixel_count in vertical_projection:
        # print(column_index + 1, pixel_count)
        if pixel_count == 0:
            if column_index < (width - 1):
                if (vertical_projection[column_index - 1] == 0) and (vertical_projection[column_index + 1] != 0):
                    horizontal_start = column_index
                if (vertical_projection[column_index - 1] != 0) and (vertical_projection[column_index + 1] == 0):
                    horizontal_end = column_index + 2
                else:
                    pass
            else:
                pass
        else:
            pass

        column_index = column_index + 1

    extract_image = original[vertical_start:vertical_end, horizontal_start:horizontal_end]

    new_height, new_width = extract_image.shape

    # break four parts from Full Image
    leftPart = extract_image[0:new_height, 0:round(new_width/2)]
    rightPart = extract_image[0:new_height, round(new_width/2):new_width]
    upperPart = extract_image[0:round(new_height/2), 0:new_width]
    downPart = extract_image[round(new_height/2):new_height, 0:new_width]

    return leftPart, rightPart, upperPart, downPart, extract_image

