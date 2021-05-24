import pandas as pd
import pickle
import sys
from NewSegmentation.NewWholeSegmenting import Char_Details
from NewSegmentation.NewWholeSegmenting import line_count
from NewSegmentation.NewWholeSegmenting import arr1
from NewSegmentation.NewWholeSegmenting import arr2


################################################################
cell_df = pd.read_csv('../CharacterClassification/CSV Files/Input_Character_CSV/SingleCharFeatureSet.csv')

pick = open('../CharacterClassification/BSavedModels/HRModel.sav', 'rb')
model = pickle.load(pick)
pick.close()

# predict the character
predicted_chars = model.predict(cell_df[0:20])
# print(character)


################################################################


word = arr2
char = arr1

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


print("char-", char)
print("word-", word)
print("WordChar-", WordChar)
print(" ")
print(" ")

star = "*"
index = 0
kat = 0
for j in range(0, len(word)):
    element = WordChar[j]
    for k in range(0, len(element)):
        no_of_char = element[k]
        kat = kat + no_of_char
        for t in range(index, kat):
            sys.stdout.write(str(predicted_chars[t]))
        index = index + no_of_char
        sys.stdout.write(" ")
    print()



