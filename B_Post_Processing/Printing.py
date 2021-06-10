import pandas as pd
import pickle
import sys
from colorama import Fore
from NewSegmentation.NewWholeSegmenting import arr1
from NewSegmentation.NewWholeSegmenting import arr2
from NewSegmentation.NewWholeSegmenting import question_slices


# Roof | James(6430) | TF(6470) | Malli | 1877(6750) | 4(6880) | 5(7000) | Malli_2 | Marshy | Issac | Lokka(7510) | New_Marshy | New_Issac |
# Baby | Baby_Error | Sales_Manager_Error | Kitchen | SM

################################################################
cell_df_new = pd.read_csv('../CharacterClassification/CSV Files/'
                          'Input_Character_CSV/Chaincode + Zone + Transition/Input_SM.csv')

pick = open('../BSavedModels/C + Z + T   Models/SVM_Model_41.sav', 'rb')
model = pickle.load(pick)
pick.close()

# predict the character
predicted_chars = model.predict(cell_df_new[0:210])


# categories of SVM Model
categories = ['o', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k',
              'l', 'm', 'n', 'p', 'r', 's', 't', 'u', 'v', 'w',
              'y', 'z', 'B', 'F', 'H', 'I', 'N', 'R', 'T', '39',
              '.', ',', 'J', 'A']


new_predicion = []

for i in range(0, len(predicted_chars)):
    index = predicted_chars[i]
    new_predicion.append(categories[index])

################################################################


# Assign arrays which import from NewWholeSegmenting file
word = arr2
char = arr1
question = question_slices



# Set No. of lines for each question
QuestionLines = []
pvot_Q = 0
index_Q = 0

for i in range(0, len(question_slices)):
    index_Q = index_Q + question_slices[i]
    newArr = []
    for j in range(pvot_Q, index_Q):
        newArr.append(word[j])
    QuestionLines.append(newArr)
    pvot_Q = pvot_Q + question_slices[i]



# set No.of words for each line
WordChar = []
pvot = 0
index = 0

for i in range(0, len(word)):
    index = index + word[i]
    newArr = []
    for j in range(pvot, index):
        newArr.append(char[j])
    WordChar.append(newArr)
    pvot = pvot + word[i]


print()

# print a Template of student Answersheet
print("Template of the Answer")
print()
star = "*"
vote = -1
newid = 0
newcat = 0
for i in range(0, len(QuestionLines)):
    item = QuestionLines[i]
    for j in range(0, len(item)):
        vote = vote + 1
        WChar = WordChar[vote]
        wordlength = item[j]
        for k in range(0, wordlength):
            sys.stdout.write(str(star*WChar[k]))
            sys.stdout.write(" ")
        print()
    print()

print()
print()



print()
print("Student Answer By the System")
print("------------------------------------------------")
print()
print()
# prin Actual results
vote = -1
newid = 0
newcat = 0
Whole_Arr = []
for i in range(0, len(QuestionLines)):
    item = QuestionLines[i]
    test1_Arr = []
    for j in range(0, len(item)):
        test2_Arr = []
        test1_Arr.append(test2_Arr)
        vote = vote + 1
        WChar = WordChar[vote]
        wordlength = item[j]
        for k in range(0, wordlength):
            test3_Arr = []
            test2_Arr.append(test3_Arr)
            newcat = newcat + WChar[k]
            for t in range(newid, newcat):
                sys.stdout.write(Fore.GREEN + str(new_predicion[t]))
                test3_Arr.append(new_predicion[t])
            newid = newid + WChar[k]
            sys.stdout.write(" ")
        print()
    print()
    Whole_Arr.append(test1_Arr)


############################################################################
"""




# print Template without considering each question
star = "*"
for j in range(0, len(word)):
    element = WordChar[j]
    for k in range(0, len(element)):
        sys.stdout.write(str(star*element[k]))
        sys.stdout.write(" ")
    print()




# print Actual result without considering each question
index = 0
kat = 0

for j in range(0, len(word)):
    element = WordChar[j]
    for k in range(0, len(element)):
        no_of_char = element[k]
        kat = kat + no_of_char
        for t in range(index, kat):
            sys.stdout.write(str(new_predicion[t]))
        index = index + no_of_char
        sys.stdout.write(" ")
    print()
"""

############################################################################
