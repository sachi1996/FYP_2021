import matplotlib.pyplot as plt
import numpy as np
import csv
from CharacterClassification.HR_Character_Features.HR_Extract_Characters import BigChar

arr = []


def append_list_as_row(filename, list_of_elem):
    # Open file in append mode
    with open(filename, 'a+', newline='') as csvfile:
        # Create a writer object from csv module
        csvwriter = csv.writer(csvfile)
        # Add field names
        csvwriter.writerow(fields)
        # Add contents of list as last row in the csv file
        csvwriter.writerows(list_of_elem)


def partition(extract_image):
    new_height, new_width = extract_image.shape

    # break four parts from Full Image
    leftPart = extract_image[0:new_height, 0:round(new_width / 2)]
    rightPart = extract_image[0:new_height, round(new_width / 2):new_width]
    upperPart = extract_image[0:round(new_height / 2), 0:new_width]
    downPart = extract_image[round(new_height / 2):new_height, 0:new_width]

    return leftPart, rightPart, upperPart, downPart


def foregroundBackround(img):
    horizontal_W_To_B = 0
    horizontal_B_To_W = 0
    vertical_W_To_B = 0
    vertical_B_To_W = 0

    img_height, img_width = img.shape

    for i in range(1, img_height):
        for j in range(1, img_width):
            if img[i][j] == 1:
                if img[i][j - 1] == 1 and img[i][j + 1] == 1 and img[i - 1][j] == 1 and img[i + 1][j] == 1:
                    break
                if img[i][j - 1] == 0:
                    horizontal_W_To_B += 1
                if img[i][j + 1] == 0:
                    horizontal_B_To_W += 1
                if img[i - 1][j] == 0:
                    vertical_W_To_B += 1
                if img[i + 1][j] == 0:
                    vertical_B_To_W += 1

    return horizontal_W_To_B, horizontal_B_To_W, vertical_W_To_B, vertical_B_To_W


def blackPixelSum(horizontalProjection):  # calculate the sum of black pixels
    count = 0
    for i in horizontalProjection:
        count += i

    return count


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
                if (img[i - 1][j] == 1 and img[i - 1][j + 1] == 1 and img[i][j + 1] == 1 and img[i + 1][j + 1] == 1
                        and img[i + 1][j] == 1 and img[i + 1][j - 1] == 1 and img[i][j - 1] == 1 and img[i - 1][
                            j - 1] == 1):
                    break
                else:
                    if img[i - 1][j] == 1:
                        direction1 += 1
                    if img[i - 1][j + 1] == 1:
                        direction2 += 1
                    if img[i][j + 1] == 1:
                        direction3 += 1
                    if img[i + 1][j + 1] == 1:
                        direction4 += 1
                    if img[i + 1][j] == 1:
                        direction5 += 1
                    if img[i + 1][j - 1] == 1:
                        direction6 += 1
                    if img[i][j - 1] == 1:
                        direction7 += 1
                    if img[i - 1][j - 1] == 1:
                        direction8 += 1

    return direction1, direction2, direction3, direction4, direction5, direction6, direction7, direction8


for t in range(0, len(BigChar)):
    char_img = BigChar[t]
    AllFeatures = []
    ChainVector = []
    ZoneVector = []
    TransVector = []


    ############################################################################### - - - - > ChainCode
    d1, d2, d3, d4, d5, d6, d7, d8 = chaincode(char_img)
    total_chain = d1 + d2 + d3 + d4 + d5 + + d6 + d7 + d8
    percentage_transition_zone_chain = 100 / total_chain

    ChainVector.append(d1 * percentage_transition_zone_chain)
    ChainVector.append(d2 * percentage_transition_zone_chain)
    ChainVector.append(d3 * percentage_transition_zone_chain)
    ChainVector.append(d4 * percentage_transition_zone_chain)
    ChainVector.append(d5 * percentage_transition_zone_chain)
    ChainVector.append(d6 * percentage_transition_zone_chain)
    ChainVector.append(d7 * percentage_transition_zone_chain)
    ChainVector.append(d8 * percentage_transition_zone_chain)


    ############################################################################### - - - - > ZoneRatio
    left, right, upper, down = partition(char_img)

    left_proj = np.sum(left, axis=1)
    right_proj = np.sum(right, axis=1)
    upper_proj = np.sum(upper, axis=1)
    down_proj = np.sum(down, axis=1)

    left_sum = blackPixelSum(left_proj)
    right_sum = blackPixelSum(right_proj)
    upper_sum = blackPixelSum(upper_proj)
    down_sum = blackPixelSum(down_proj)

    total_Zone = left_sum + right_sum + upper_sum + down_sum
    percentage_transition_zone = (100 / total_Zone)

    ZoneVector.append(left_sum * percentage_transition_zone)
    ZoneVector.append(right_sum * percentage_transition_zone)
    ZoneVector.append(upper_sum * percentage_transition_zone)
    ZoneVector.append(down_sum * percentage_transition_zone)


    ############################################################################### - - - - > Transition
    hf, hb, vf, vb = foregroundBackround(char_img)

    total_transition = hf + hb + vf + vb
    percentage_transition = 0

    if total_transition != 0:
        percentage_transition = 100 / total_transition

    TransVector.append(hf * percentage_transition)
    TransVector.append(hb * percentage_transition)
    TransVector.append(vf * percentage_transition)
    TransVector.append(vb * percentage_transition)


    ############################################################################### - - - - > Create Full Feature Array
    AllFeatures = ChainVector
    for j in range(len(ZoneVector)):
        AllFeatures.append(ZoneVector[j])
    for k in range(len(TransVector)):
        AllFeatures.append(TransVector[k])

    arr.append(AllFeatures)


character_data = ['1', 'dot', 'a', 'dot', 'T', 'b', 'dot', 'F', 'c', 'dot', 'T', 'd', 'dot', 'F',
                  '2', 'dot', 'e', 'n', 'v', 'i', 'r', 'O', 'n', 'm', 'a', 'n', 't', 'a', 'l',
                  'f', 'r', 'i', 'e', 'n', 'd', 'l', 'y', '3', 'dot', 'B', 'e', 'c', 'a', 'u', 's', 'e',
                  'i', 't', 'r', 'a', 'i', 'n', 'e', 'd', 's', 'O', 'h', 'a', 'r', 'd',
                  '4', 'dot', 'N', 'u', 'r', 's', 'e', 'r', 'y', 'R', 'O', 'O', 'm', '5', 'dot',
                  'R', 'O', 'O', 'f', 'h', 'a', 'v', 'e', 'a', 't', 'r', 'e', 'e', 'O', 'n', 't', 'O', 'p',
                  'O', 'f', 'i', 't', '6', 'dot', 'b', '7', 'dot', '1', 'dot',
                  'B', 'u', 'i', 'l', 'd', 'e', 'r', 's', '2', 'dot', 's', 'c', 'h', 'O', 'O', 'l']


labels = [1, 40, 10, 40, 38, 11, 40, 33, 12, 40, 38, 13, 40, 33, 2, 40, 14, 22, 28, 18, 24, 0, 22, 21,
          10, 22, 26, 10, 20, 15, 24, 18, 14, 22, 13, 20, 30, 3, 40, 32, 14, 12, 10, 27, 25, 14, 18, 26,
          24, 10, 18, 22, 14, 13, 25, 0, 17, 10, 24, 13, 4, 40, 36, 27, 24, 25, 14, 24, 30, 37, 0, 0, 21,
          5, 40, 37, 0, 0, 15, 17, 10, 28, 14, 10, 26, 24 ,14, 14, 0, 22, 26, 0, 23, 0, 15, 18, 26,
          6, 40, 11, 7, 40, 1, 40, 32, 27, 18, 20, 13, 14, 24, 25, 2, 40, 25, 12, 17, 0, 0, 20]


for i in range(0, len(character_data)):
    print(str(character_data[i]) + " : " + str(labels[i]))


    ############################################################################### - - - - > Create CSV File
fields = ['target_names', 'direction1',
                          'direction2',
                          'direction3',
                          'direction4',
                          'direction5',
                          'direction6',
                          'direction7',
                          'direction8',
                          'left_zone',
                          'right_zone',
                          'upper_zone',
                          'down_zone',
                          'horizontal_w_to_b',
                          'horizontal_b_to_w',
                          'vertical_w_to_b',
                          'vertical_b_to_w','target']

# data rows of csv file
rows = []


print()
print()
print("Length of arr : " + str(len(arr)))
print("Length of CharacterData : " + str(len(character_data)))
print("Length of labels : " + str(len(labels)))

for i in range(len(arr)):
    features = arr[i]
    row = [character_data[i],
           features[0],
           features[1],
           features[2],
           features[3],
           features[4],
           features[5],
           features[6],
           features[7],
           features[8],
           features[9],
           features[10],
           features[11],
           features[12],
           features[13],
           features[14],
           features[15],
           labels[i]]
    rows.append(row)

filename = "F:/Python/OpenCV/CharacterClassification/CSV Files/New_All_Characters/All_Characters.csv"


def append_list_as_row(filename, list_of_elem):
    # Open file in append mode
    with open(filename, 'a+', newline='') as csvfile:
        # Create a writer object from csv module
        csvwriter = csv.writer(csvfile)
        # Add field names
        # csvwriter.writerow(fields)
        # Add contents of list as last row in the csv file
        csvwriter.writerows(list_of_elem)


append_list_as_row(filename, rows)

