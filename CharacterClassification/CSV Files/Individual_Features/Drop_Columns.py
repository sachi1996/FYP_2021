import csv
import pandas as pd
from colorama import Fore

# a = pd.read_csv("../Seperate_Answer_Sheets/Chaincode/Roof_James.csv")

"""
input_csv = '../Seperate_Answer_Sheets/Chaincode + Zone + Transition/All_James.csv'
output_csv = '../Seperate_Answer_Sheets/Transition/James.csv'


with open(input_csv, "r") as source:
    reader = csv.reader(source)
    with open(output_csv, "w") as result:
        writer = csv.writer(result)
        for colmn in reader:
            writer.writerow((colmn[0], colmn[13], colmn[14], colmn[15], colmn[16], colmn[17]))
"""

print(Fore.RED + str("Hello Red"))
print(Fore.GREEN + str("Hello Green"))
