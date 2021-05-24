import cv2
import matplotlib.pyplot as plt
import numpy as np

inputImage = 'F:/Final year project/Test Images/Light.jpg'
camScanner = cv2.imread(inputImage)
grayCamScanner = cv2.cvtColor(camScanner, cv2.COLOR_BGR2GRAY)
GaussianFiltered5 = cv2.GaussianBlur(grayCamScanner, (5, 5), 0)
_, GaussianThresh5 = cv2.threshold(GaussianFiltered5, 127, 255, cv2.THRESH_BINARY)

original = GaussianThresh5.copy()

height, width = original.shape

original[original == 0] = 1
original[original == 255] = 0

horizontal_projection = np.sum(original, 1)

row_index = 0
top_edge = []
bottom_edge = []

for pixel_count in horizontal_projection:
    # print(row_index + 1, pixel_count)
    if pixel_count == 0:
        if row_index < (height - 1):
            if (horizontal_projection[row_index - 1] == 0) and (horizontal_projection[row_index + 1] != 0):
                top_edge.append(row_index - 1)
            if (horizontal_projection[row_index - 1] != 0) and (horizontal_projection[row_index + 1] == 0):
                bottom_edge.append(row_index + 1)
            else:
                pass
        else:
            pass
    else:
        pass

    row_index = row_index + 1

print(top_edge)
print(bottom_edge)
