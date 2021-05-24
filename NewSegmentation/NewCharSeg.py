import numpy as np


def char_segment(img, char_start, char_end):
    h, w = img.shape
    projection = np.sum(img, axis=0)

    row_index = 0
    for pixel_count in projection:
        if pixel_count == 0:
            if row_index < (w - 1):
                if (projection[row_index - 1] == 0) and (projection[row_index + 1] != 0):
                    char_start.append(row_index - 1)
                if (projection[row_index - 1] != 0) and (projection[row_index + 1] == 0):
                    char_end.append(row_index + 2)
                else:
                    pass
            else:
                pass
        else:
            pass

        row_index = row_index + 1

