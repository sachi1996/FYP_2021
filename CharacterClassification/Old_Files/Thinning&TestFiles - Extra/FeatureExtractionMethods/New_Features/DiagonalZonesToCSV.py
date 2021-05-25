import csv
from DiagonalZones import BigZoneDensity
from csv import writer

zone_features = BigZoneDensity

# Combine Features Storing
image_set = zone_features

fields = ['target_names', 'left_zone',
                          'right_zone',
                          'upper_zone',
                          'down_zone',
                          'target']

# data rows of csv file
rows = []

for i in range(len(image_set)):
    features = image_set[i]
    row = ['y',
           features[0],
           features[1],
           features[2],
           features[3],
           25]
    rows.append(row)

#####################################################################################################

# name of csv file

filename = "F:/Python/OpenCV/CharacterClassification/CSV Files/my_zone.csv"


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


