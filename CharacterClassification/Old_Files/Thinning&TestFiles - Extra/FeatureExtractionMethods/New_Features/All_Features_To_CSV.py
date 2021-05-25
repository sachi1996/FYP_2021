import csv
from ChainCode import BigChainCode
from DiagonalZones import BigZoneDensity
from ForeBack import BigForeBackground
from csv import writer


chaincode_features = BigChainCode
zone_features = BigZoneDensity
foreback_features = BigForeBackground

CombineFeatures = []

for i in range(len(BigChainCode)):
    arr = BigChainCode[i]
    for j in range(len(BigZoneDensity[0])):
        arr.append(BigZoneDensity[i][j])
    for k in range(len(BigForeBackground[0])):
        arr.append(BigForeBackground[i][k])
    CombineFeatures.append(arr)

print(CombineFeatures)


# Combine Features Storing
image_set = CombineFeatures

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

for i in range(len(image_set)):
    features = image_set[i]
    row = ['T',
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
           20]
    rows.append(row)

#####################################################################################################

# name of csv file

filename = "F:/Python/OpenCV/CharacterClassification/CSV Files/Combined_CSV/Combined_Features.csv"


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


