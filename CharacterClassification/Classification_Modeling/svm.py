import csv
from _ast import mod

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
from sklearn import metrics

cell_df = pd.read_csv('../CSV Files/Combined_CSV/Latest_Generated_Features.csv')
# Test_Cell = pd.read_csv('../CSV Files/TestCodes/ChainCodeTest.csv')

df_k = cell_df[cell_df.target == 13]
df_z = cell_df[cell_df.target == 26]
df_y = cell_df[cell_df.target == 25]


# show claassification of images
# plt.scatter(df_k['direction1'], df_k['direction2'], color='blue', marker="s")
# plt.scatter(df_z['direction1'], df_z['direction2'], color='red', marker='.')
# plt.scatter(df_y['direction1'], df_y['direction2'], color='green', marker='+')
# plt.show()


X = cell_df.drop(['target_names', 'target'], axis='columns')
y = cell_df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print("Full Size : " + str(len(X_train) + len(X_test)))
print("Train Size : " + str(len(X_train)))
print("Test Size : " + str(len(X_test)))
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
print("\n\n")

model = SVC(kernel="poly", C=200)
model.fit(X_train, y_train)

#####################################################################################
# prediction = model.predict(X_test)
# accuracy = model.score(X_test, y_test)
#
categories = ['0', '1', '2', '3', '4',
              '5', 'F', '7', '8', '9',
              '10', '11', '12', 'k', '14',
              '15', '16', '17', '18', '19',
              '20', '21', '22', '23', '24', 'y', 'z']
#
# print("Accuracy : " + str(accuracy))
# print("Prediction : ", categories[prediction[0]])
#
# myCharacter = cv2.imread(X_test[0])
# name = plt.imshow(myCharacter)
# plt.show()
#####################################################################################


accuracy = model.score(X_test, y_test)
# prediction = model.predict(X_test)
# print("Prediction Accuracy : ",metrics.accuracy_score(y_test, prediction))

# print(classification_report(y_test, prediction))

# columnsData = cell_df.loc[:, 'direction1']
# #
# print(index)
# print(columnsData[index-2])

s = pickle.dumps(model)
clf = pickle.loads(s)
print("First Test 10 rows")
print(X_test[0:10])
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
print("\n\n\n")

first_five = clf.predict(X_test[0:5])
print("Prediction of First Five")
print(first_five)
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
print("\n\n\n")

"""
print("Test_Cell")
print(Test_Cell)
prd = clf.predict(Test_Cell[0:1])
print(prd)
"""

print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")

"""
arr = []
for i in range(0, 10):
    index = abc[i]
    arr.append(categories[index])
"""
# print(arr)

print("Your Accuracy : ", round(accuracy*100, 2), "%")
print("")
print("Done Model")

