from itertools import chain

import pandas as pd
import pickle
import sys
from NewSegmentation.NewWholeSegmenting import Char_Details
from NewSegmentation.NewWholeSegmenting import line_count
from NewSegmentation.NewWholeSegmenting import word_count
from NewSegmentation.NewWholeSegmenting import char_count

cell_df = pd.read_csv('../CharacterClassification/CSV Files/Input_Character_CSV/SingleCharFeatureSet.csv')

pick = open('../CharacterClassification/BSavedModels/HRModel.sav', 'rb')
model = pickle.load(pick)
pick.close()


a = 2
b = 9
# predict the character
character = model.predict(cell_df[0:20])

items = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333, 444, 555, 666, 777, 888, 999]

print(character[1])






