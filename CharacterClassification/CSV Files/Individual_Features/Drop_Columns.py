import csv
import pandas as pd

# a = pd.read_csv("../Input_Character_CSV/Chaincode/Input_Roof.csv")

input_csv = '../Input_Character_CSV/Chaincode + Zone + Transition/Input_Lokka.csv'
output_csv = '../Input_Character_CSV/Transition/Input_Lokka.csv'


with open(input_csv, "r") as source:
    reader = csv.reader(source)
    with open(output_csv, "w") as result:
        writer = csv.writer(result)
        for colmn in reader:
            writer.writerow((colmn[12], colmn[13], colmn[14], colmn[15]))


