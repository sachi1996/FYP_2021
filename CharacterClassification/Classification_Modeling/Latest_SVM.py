from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import matplotlib.pyplot as plt
cell_df = pd.read_csv('../CSV Files/Seperate_Answer_Sheets'
                      '/Chaincode + Zone + Transition/All_File.csv')

X = cell_df.drop(['target_names', 'target'], axis='columns')
y = cell_df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=209)


# generate the model
cls = svm.SVC(kernel="poly", gamma=0.001, C=400)

# train the model
cls.fit(X_train, y_train)

predict = cls.predict(X_test)
accuracy = cls.score(X_test, y_test)


pick = open(
    '../../BSavedModels/C + Z + T   Models/SVM_Model_10.sav', 'wb')
pickle.dump(cls, pick)
pick.close()


# accuracy
Accuracy = metrics.accuracy_score(y_test, y_pred=predict)
print("Prediction Accuracy : ", round(Accuracy*100, 2))


