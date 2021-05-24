class Computer:

    def __init__(self, getLine, getWord, getChar, getImg):
        self.line = getLine
        self.word = getWord
        self.char = getChar
        self.img = getImg

    def config(self):
        print("Line : " + str(self.line) + ", Word : " + str(self.word) + ", Char : " + str(self.char) + ", Img - " + str(self.img))


com1 = Computer(1, 2, 4, "img1")
com2 = Computer(2, 3, 5, "img2")


arr = [[1, 1, 1, "img1"], [1, 1, 2, "img2"], [1, 2, 2, "img3"], [2, 3, 4, "img4"], [3, 3, 2, "img5"], [4, 3, 1, "img6"]]


Char_Details = []
person = {
    'line': 1,
    'word': 1,
    'char': 1,
    'img': "img1"
}

for sub_array in arr:
    person['line'] = sub_array[0]
    person['word'] = sub_array[1]
    person['char'] = sub_array[2]
    person['img'] = sub_array[3]
    Char_Details.append(person)

print(Char_Details)


