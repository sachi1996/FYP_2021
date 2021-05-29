from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import matplotlib.pyplot as plt
cell_df = pd.read_csv('../CSV Files/HR_Features/All_Char_A_J_James.csv')

X = cell_df.drop(['target_names', 'target'], axis='columns')
y = cell_df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=209)


# generate the model
cls = svm.SVC(kernel="poly", gamma=0.001, C=100)

# train the model
cls.fit(X_train, y_train)

# predict the response
predict = cls.predict(X_test)
accuracy = cls.score(X_test, y_test)


pick = open('F:/Python/OpenCV/BSavedModels/Input_James.sav', 'wb')
pickle.dump(cls, pick)
pick.close()


# accuracy
Accuracy = metrics.accuracy_score(y_test, y_pred=predict)
print("Prediction Accuracy : ", round(Accuracy*100, 2))


















"""
##########################################################################

cell_df_new = pd.read_csv('../CSV Files/Input_Character_CSV/SingleCharFeatureSet.csv')

predicted_chars = cls.predict(cell_df_new[0:27])

# print(character)
print("Predicted Chars", predicted_chars)
print("Length of Predicted Chars", len(predicted_chars))



# New Generated CSV
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'm', 'n', 'p', 'r', 's', 't', 'u', 'w', 'y', 'z', 'B', 'F', 'H', 'T']


new_predicion = []

for i in range(0, len(predicted_chars)):
    index = predicted_chars[i]
    new_predicion.append(categories[index])

print("New Prediction", new_predicion)

########################################################################
"""




















"""
for i in range(20, 1000, 10):
    cls = svm.SVC(kernel="poly", C=i)
    cls.fit(X_train, y_train)
    # predict = cls.predict(X_test)
    # Accuracy = metrics.accuracy_score(y_test, y_pred=predict)
    accuracy = cls.score(X_test, y_test)
    print("C - " + str(i) + " >>>> Accuracy : " + str(accuracy))
"""

"""
# precision score
print("Precision : ", metrics.precision_score(y_test, y_pred=predict))
print(" ")

# recall score
print("Recall : ", metrics.recall_score(y_test, y_pred=predict))
print(" ")

# classification report
print(metrics.classification_report(y_test, y_pred=predict))
print(" ")
"""



"""
pick = open('F:/Python/OpenCV/CharacterClassification/BSavedModels/HRModel4.sav', 'wb')
pickle.dump(cls, pick)
pick.close()
"""