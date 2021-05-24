import pandas as pd
import pickle
import sys
from NewSegmentation.NewWholeSegmenting import arr1
from NewSegmentation.NewWholeSegmenting import arr2

################################################################
cell_df_new = pd.read_csv('../CharacterClassification/CSV Files/Input_Character_CSV/Input_rmophl.csv')

pick = open('../CharacterClassification/BSavedModels/rmophl.sav', 'rb')
model = pickle.load(pick)
pick.close()

# predict the character
predicted_chars = model.predict(cell_df_new[0:119])
# print(character)

print("Predicted Chars", predicted_chars)
print("Length of Predicted Chars", len(predicted_chars))


categories = ['o', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a',
              'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'm',
              'n', 'p', 'r', 's', 't', 'u', 'w', 'y', 'z', 'B',
              'F', 'H', 'T', 'I', '35', '36', '37', '38', '39', '.']


new_predicion = []

for i in range(0, len(predicted_chars)):
    index = predicted_chars[i]
    new_predicion.append(categories[index])

print("New Prediction", new_predicion)
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


print(" ")
print(" ")


star = "*"
for j in range(0, len(word)):
    element = WordChar[j]
    for k in range(0, len(element)):
        sys.stdout.write(str(star*element[k]))
        sys.stdout.write(" ")
    print()


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

