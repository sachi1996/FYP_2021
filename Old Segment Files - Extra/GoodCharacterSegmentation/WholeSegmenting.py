import cv2
import matplotlib.pyplot as plt
import numpy as np
from LineSegmenting import projection_segment
from WordSegmenting import word_segment

inputImage = 'F:/Final year project/Test Images/Light.jpg'
fullImage = cv2.imread(inputImage)
plt.imshow(fullImage, cmap="gray")
# plt.show()

BigChar = []
char_line = []

###########################################################################################


class CharClass:
    def config(self):
        print("15, 16gb, 1TB")


com1 = CharClass()

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
word_start = []
word_end = []
char_start = []
char_end = []

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
    # plt.show()

    # vericle projection is taken
    verticle_projection = np.sum(img1, 0)
    char_start.clear()  # clear previous appended values
    char_end.clear() # clear previous appended values
    char_slices.clear() # clear previous appended values
    word_start.clear() # clear previous appended values
    word_end.clear() # clear previous appended values
    word_slices.clear() # clear previous appended values
    projection_segment(verticle_projection, row_index, width, char_start, char_end) # char start and ends taken
    word_start = [char_start[0]]
    word_end = []
    word_segment(char_start, char_end, word_start, word_end) # word start and ends taken

    # append and show word slices
    for t in range(0, len(word_start)):
        word_slices.append(img1[0:line_height, word_start[t]:word_end[t]])
        name = "Word - " + str(t + 1)
        plt.imshow(word_slices[t], cmap='gray')
        plt.title(name)
        # plt.show()

    # append and show character slices
    for j in range(0, len(char_start)):
        char_slices.append(img1[0:line_height, char_start[j]:char_end[j]])
        char_name = "Character - " + str(j + 1)
        char_img = char_slices[j]
        plt.imshow(char_img, cmap="gray")

        word_number = 0
        for i in range(0, len(word_end)):
            if (char_start[j] < word_end[i]) and (char_start[j] >= word_start[i]):
                word_number = i+1
            else:
                pass

        print("line-" + str(line_num) + ", Word-" + str(word_number))
        plt.show()

        new_height, new_width = char_img.shape
        h_proj = np.sum(char_img, 1)

        index = 0
        v_start = 0
        v_end = 0

        for pixel_count in h_proj:
            # print(index + 1, pixel_count)
            if pixel_count == 0:
                if index < (new_height - 1):
                    if (h_proj[index - 1] == 0) and (h_proj[index + 1] != 0):
                        v_start = index - 1
                    if (h_proj[index - 1] != 0) and (h_proj[index + 1] == 0):
                        v_end = index + 2
                    else:
                        pass
                else:
                    pass
            else:
                pass

            index = index + 1


        # print("V-Start : " + str(v_start))
        # print("V-End : " + str(v_end))
        extract_char = char_img[v_start:v_end, 0:new_width]



    BigChar.append(char_line)

cv2.waitKey(0)
cv2.destroyAllWindows()


