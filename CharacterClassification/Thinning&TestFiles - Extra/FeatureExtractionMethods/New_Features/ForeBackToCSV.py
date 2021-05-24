import csv
from ForegroundBackground import BigForeBackground
from csv import writer

foreback_features = BigForeBackground

# Combine Features Storing
image_set = foreback_features

fields = ['target_names', 'horizontal_w_to_b',
                          'horizontal_b_to_w',
                          'vertical_w_to_b',
                          'vertical_b_to_w',
                          'target']

# data rows of csv file
rows = []

for i in range(len(image_set)):
    features = image_set[i]
    row = ['m',
           features[0],
           features[1],
           features[2],
           features[3],
           13]
    rows.append(row)

#####################################################################################################

# name of csv file

filename = "F:/Python/OpenCV/CharacterClassification/CSV Files/my_foreback.csv"


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


