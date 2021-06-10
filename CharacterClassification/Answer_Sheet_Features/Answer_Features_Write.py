import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import pickle
from NewSegmentation.NewWholeSegmenting import Char_Details


filename = '../CSV Files/Input_Character_CSV/Chaincode + Zone + Transition/Test_Issac.csv'


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
    leftPart = extract_image[0:new_height, 0:round(new_width/2)]
    rightPart = extract_image[0:new_height, round(new_width/2):new_width]
    upperPart = extract_image[0:round(new_height/2), 0:new_width]
    downPart = extract_image[round(new_height/2):new_height, 0:new_width]
    
    return leftPart,rightPart, upperPart, downPart


def foregroundBackround(img):

    horizontal_W_To_B = 0
    horizontal_B_To_W = 0
    vertical_W_To_B = 0
    vertical_B_To_W = 0

    img_height, img_width = img.shape

    for i in range(1, img_height):
        for j in range(1, img_width):
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


BigArr = []

for t in range(0, len(Char_Details)):
    char_img = Char_Details[t]['img']
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
    percentage_transition_zone = (100/total_Zone)
    
    ZoneVector.append(left_sum * percentage_transition_zone)
    ZoneVector.append(right_sum * percentage_transition_zone)
    ZoneVector.append(upper_sum * percentage_transition_zone)
    ZoneVector.append(down_sum * percentage_transition_zone)
    
    ############################################################################### - - - - > Transition
    hf, hb, vf, vb = foregroundBackround(char_img)

    total_transition = hf + hb + vf + vb
    percentage_transition = 0
    
    if total_transition != 0:
        percentage_transition = 100/total_transition

    TransVector.append(hf*percentage_transition)
    TransVector.append(hb*percentage_transition)
    TransVector.append(vf*percentage_transition)
    TransVector.append(vb*percentage_transition)

    ############################################################################### - - - - > Create Full Feature Array
    AllFeatures = ChainVector
    for j in range(len(ZoneVector)):
        AllFeatures.append(ZoneVector[j])
    for k in range(len(TransVector)):
        AllFeatures.append(TransVector[k])

    print(AllFeatures)

    ############################################################################### - - - - > Create CSV File
    fields = ['direction1',
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
              'vertical_b_to_w']

    features = AllFeatures

    row = [features[0],
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
           features[15],]
    BigArr.append(row)

# print(BigArr)

append_list_as_row(filename, BigArr)



