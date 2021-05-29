####################################### 01-
"""
character_data = ['O', '1', 'dot', 'a', 'dot', 'T', 'r', 'u', 'e', 'b', 'dot', 'F', 'a', 'l',
                  's', 'e', 'c', 'dot', 'T', 'r', 'u', 'e', 'd', 'dot', 'F', 'a', 'l', 's',
                  'e', 'O', '2', 'dot', 'I', 't', 'i', 's', 'g', 'O', 'O', 'd', 'd', 'e', 'c',
                  'i', 's', 'i', 'O', 'n', 'O', '3', 'dot', 'B', 'e', 'c', 'a', 'u', 's', 'e',
                  'h', 'e', 'h', 'a', 's', 'b', 'e', 't', 't', 'e', 'r', 'c', 'O', 'n', 'n',
                  'e', 'c', 't', 'i', 'O', 'n', 'b', 'e', 't', 'w', 'e', 'e', 'n', 'h', 'i',
                  's', 'f', 'r', 'i', 'e', 'n', 'd', 's', 'dot', 'O', '4', 'dot', 'I', 'n',
                  't', 'e', 'r', 'n', 'e', 't', 'z', 'O', 'O', 'm', 'y', 'O', 'u', 't', 'u',
                  'b', 'e']

labels = [0, 1, 40, 10, 40, 33, 23, 26, 14, 11, 40, 31, 10, 19, 24, 14,
          12, 40, 33, 23, 26, 14, 13, 40, 31, 10, 19, 24, 14, 0, 2, 40,
          34, 25, 18, 24, 16, 0, 0, 13, 13, 14, 12, 18, 24, 18, 0, 21,
          0, 3, 40, 30, 14, 12, 10, 26, 24, 14, 17, 14, 17, 10, 24, 11,
          14, 25, 25, 14, 23, 12, 0, 21, 21, 14, 12, 25, 18, 0, 21, 11,
          14, 25, 27, 14, 14, 21, 17, 18, 24, 15, 23, 18, 14, 21, 13, 24,
          40, 0, 4, 40, 34, 21, 25, 14, 23, 21, 14, 25, 29 ,0, 0, 20, 28, 0,
          26, 25, 26, 11, 14]

categories = ['o', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a',
              'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'm',
              'n', 'p', 'r', 's', 't', 'u', 'w', 'y', 'z', 'B',
              'F', 'H', 'T', 'I', '35', '36', '37', '38', '39', '.']
"""


"""
#########################################################################################
categories = ['o', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k',
              'l', 'm', 'n', 'p', 'r', 's', 't', 'u', 'v', 'w',
              'y', 'z', 'B', 'F', 'H', 'I', 'N', 'R', 'T', '39', '.']
"""



"""
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Roof
character_data = ['1', 'dot', 'a', 'dot', 'T', 'b', 'dot', 'F', 'c', 'dot', 'T', 'd', 'dot', 'F',
                  '2', 'dot', 'e', 'n', 'v', 'i', 'r', 'O', 'n', 'm', 'a', 'n', 't', 'a', 'l',
                  'f', 'r', 'i', 'e', 'n', 'd', 'l', 'y', '3', 'dot', 'B', 'e', 'c', 'a', 'u', 's', 'e',
                  'i', 't', 'r', 'a', 'i', 'n', 'e', 'd', 's', 'O', 'h', 'a', 'r', 'd',
                  '4', 'dot', 'N', 'u', 'r', 's', 'e', 'r', 'y', 'R', 'O', 'O', 'm', '5', 'dot',
                  'R', 'O', 'O', 'f', 'h', 'a', 'v', 'e', 'a', 't', 'r', 'e', 'e', 'O', 'n', 't', 'O', 'p',
                  'O', 'f', 'i', 't', '6', 'dot', 'b', '7', 'dot', '1', 'dot',
                  'B', 'u', 'i', 'l', 'd', 'e', 'r', 's', '2', 'dot', 's', 'c', 'h', 'O', 'O', 'l']


labels = [1, 40, 10, 40, 38, 11, 40, 33, 12, 40, 38, 13, 40, 33, 2, 40, 14, 22, 28, 18, 24, 0, 22, 21,
          10, 22, 26, 10, 20, 15, 24, 18, 14, 22, 13, 20, 30, 3, 40, 32, 14, 12, 10, 27, 25, 14, 18, 26,
          24, 10, 18, 22, 14, 13, 25, 0, 17, 10, 24, 13, 4, 40, 36, 27, 24, 25, 14, 24, 30, 37, 0, 0, 21,
          5, 40, 37, 0, 0, 15, 17, 10, 28, 14, 10, 26, 24 ,14, 14, 0, 22, 26, 0, 23, 0, 15, 18, 26,
          6, 40, 11, 7, 40, 1, 40, 32, 27, 18, 20, 13, 14, 24, 25, 2, 40, 25, 12, 17, 0, 0, 20]

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
