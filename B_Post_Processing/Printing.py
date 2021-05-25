import pandas as pd
import pickle
import sys
from NewSegmentation.NewWholeSegmenting import arr1
from NewSegmentation.NewWholeSegmenting import arr2
from NewSegmentation.NewWholeSegmenting import question_slices

################################################################
cell_df_new = pd.read_csv('../CharacterClassification/CSV Files/Input_Character_CSV/Input_Roof.csv')

pick = open('../BSavedModels/Input_Roof6.sav', 'rb')
model = pickle.load(pick)
pick.close()

# predict the character
predicted_chars = model.predict(cell_df_new[0:120])


# categories of SVM Model
categories = ['o', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k',
              'l', 'm', 'n', 'p', 'r', 's', 't', 'u', 'v', 'w',
              'y', 'z', 'B', 'F', 'H', 'I', 'N', 'R', 'T', '39', '.']


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
print("Real Answer")
print()
# prin Actual results
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
            newcat = newcat + WChar[k]
            for t in range(newid, newcat):
                sys.stdout.write(str(new_predicion[t]))
            newid = newid + WChar[k]
            sys.stdout.write(" ")
        print()
    print()


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
