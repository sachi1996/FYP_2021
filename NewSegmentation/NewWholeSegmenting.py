import cv2
import matplotlib.pyplot as plt
import numpy as np
from NewSegmentation.NewLineSeg import projection_segment
from NewSegmentation.NewWordSeg import word_segment
from NewSegmentation.NewCharSeg import char_segment

inputImage = 'G:/imwrite images/Given Images/Issac.jpg'
fullImage = cv2.imread(inputImage)
plt.imshow(fullImage, cmap="gray")
plt.show()

BigChar = []
char_line = []

###########################################################################################

# get all the character details 
Char_Details = []

line_count = 0
word_count = 0
char_count = 0

arr1 = []
arr2 = []
questionArr = []
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
word_character_start = []
word_character_end = []

projection_segment(horizontal_projection, row_index, height, line_start, line_end)

###########################################################################
# get how many lines along with each question
line_differences = []

for i in range(0, len(line_end)):
    if i == (len(line_end)-1):
        break
    else:
        line_differences.append(line_start[i+1]-line_end[i])

print("Line Differences : ", line_differences)

question_slices = []
demo_arr = [0]
pvot = 0

line_Val = 0
for i in line_differences:
    line_Val = line_Val + i

Avg_Difference = (line_Val/len(line_differences)) - 5

for i in range(0, len(line_differences)):
    if line_differences[i] > Avg_Difference:
        demo_arr.append(i+1)

demo_arr.append(len(line_start))


for i in range(0, len(demo_arr)):
    count = i + 1
    if count == len(demo_arr):
        break
    else:
        question_slices.append(demo_arr[count]-demo_arr[i])


###########################################################################


line_slices = []
word_slices = []
char_slices = []

# print("Lengh of Lines : " + str(len(line_start)))
line_num = 0
for x in range(len(line_start)):
    line_count = line_count + 1
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
    word_start.clear() # clear previous appended values
    word_end.clear() # clear previous appended values
    word_slices.clear() # clear previous appended values
    char_start.clear()  # clear previous appended values
    char_end.clear() # clear previous appended values
    projection_segment(verticle_projection, row_index, width, char_start, char_end) # char start and ends taken
    word_start = [char_start[0]]
    word_end = []
    word_segment(char_start, char_end, word_start, word_end) # word start and ends taken
    # char_segment(img1, character_start, character_end,  word_start, word_end)

    # append and show word slices
    for t in range(0, len(word_start)):
        word_count = word_count + 1
        word_character_start.clear()  # clear previous appended values
        word_character_end.clear()  # clear previous appended values
        char_slices.clear()  # clear previous appended values

        word_slices.append(img1[0:line_height, word_start[t]:word_end[t]])
        name = "Word - " + str(t + 1)
        word_image = word_slices[t]
        plt.imshow(word_image, cmap='gray')
        plt.title(name)
        # plt.show()

        char_segment(word_image, word_character_start, word_character_end)

        for k in range(0, len(word_character_start)):
            char_count = char_count + 1
            char_slices.append(word_image[0:line_height, word_character_start[k]:word_character_end[k]])
            name = "Char - " + str(k + 1)
            char_image = char_slices[k]
            plt.imshow(char_image, cmap='gray')
            plt.title(name)
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

            # print("V-Start : " + str(v_start))
            # print("V-End : " + str(v_end))
            extract_char = char_image[v_start:v_end, 0:new_width]
            plt.imshow(extract_char, cmap="gray")
            cv2.imwrite("G:/imwrite images/imwrite/line" + str(x+1) + "-word" + str(t+1) + "-char" + str(k+1) + ".jpg", extract_char)
            # plt.show()

            person = {
                'line': 0,
                'word': 0,
                'char': 0,
                'img': fullImage
            }

            person['line'] = x+1
            person['word'] = t+1
            person['char'] = k+1
            person['img'] = extract_char
            Char_Details.append(person)

        arr1.append(len(char_slices))
    arr2.append(len(word_slices))


cv2.waitKey(0)
cv2.destroyAllWindows()



