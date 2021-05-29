import cv2
import matplotlib.pyplot as plt
import numpy as np
from NewSegmentation.NewLineSeg import projection_segment
from NewSegmentation.NewWordSeg import word_segment
from NewSegmentation.NewCharSeg import char_segment

inputImage = 'G:/imwrite images/Given Images/Demo.jpg'
fullImage = cv2.imread(inputImage)
plt.imshow(fullImage, cmap="gray")
plt.show()

BigChar = []
char_line = []

###########################################################################################

# get all the character details
Char_Details = []

###########################################################################################
# Apply preProcessing for input image

camScanner = cv2.imread(inputImage)
grayCamScanner = cv2.cvtColor(camScanner, cv2.COLOR_BGR2GRAY)
GaussianFiltered5 = cv2.GaussianBlur(grayCamScanner, (5, 5), 0)
_, GaussianThresh5 = cv2.threshold(GaussianFiltered5, 127, 255, cv2.THRESH_BINARY)

original = GaussianThresh5.copy()

height, width = original.shape

original[original == 0] = 1
original[original == 255] = 0

horizontal_projection = np.sum(original, 1)


#######################################################################################################
# Apply line segmnentation

row_index = 0
line_start = []
line_end = []


projection_segment(horizontal_projection, row_index, height, line_start, line_end)

line_slices = []
word_slices = []
char_slices = []

# print("Lengh of Lines : " + str(len(line_start)))
line_num = 0
for x in range(len(line_start)):
    line_num = line_num + 1
    line_slices.append(original[line_start[x]:line_end[x], 0:width])
    img1 = line_slices[x]
    line_height, line_width = img1.shape
    line_name = "Line - " + str(x + 1)
    plt.imshow(line_slices[x], cmap='gray')
    plt.title(line_name)
    # cv2.imwrite("G:/imwrite images/"+str(line_name)+".jpg", img1)
    # plt.show()

    # vericle projection is taken
    verticle_projection = np.sum(img1, 0)
    char_start = []
    char_end = []
    char_slices.clear()
    projection_segment(verticle_projection, row_index, width, char_start, char_end) # char start and ends taken

    for k in range(0, len(char_start)):
        char_slices.append(img1[0:line_height, char_start[k]:char_end[k]])
        char_name = "Char - " + str(k + 1)
        char_image = char_slices[k]
        plt.imshow(char_image, cmap='gray')
        # plt.show()

        new_height, new_width = char_image.shape
        h_proj = np.sum(char_image, 1)

        index = 0
        v_start = 0
        v_end = 0

        for pixel_count in h_proj:
            # print(index + 1, pixel_count)
            if pixel_count == 0:
                if index < (new_height - 1):
                    if (h_proj[index - 1] == 0) and (h_proj[index + 1] != 0):
                        if v_start == 0:
                            v_start = index - 1
                        else:
                            pass
                    if (h_proj[index - 1] != 0) and (h_proj[index + 1] == 0):
                        v_end = index + 2
                    else:
                        pass
                else:
                    pass
            else:
                pass

            index = index + 1

        extract_char = char_image[v_start:v_end, 0:new_width]
        plt.imshow(extract_char, cmap='gray')
        plt.title("line-" + str(x+1) + " : char-" + str(k+1))
        # cv2.imwrite("G:/imwrite images/"+str(line_name)+".jpg", img1)
        # plt.show()
        BigChar.append(extract_char)

print("No.of Characters - " + str(len(BigChar)))

cv2.waitKey(0)
cv2.destroyAllWindows()

