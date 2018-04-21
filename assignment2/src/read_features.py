import csv

import numpy as np
from scipy import stats


def extract_features(csv_file=None):
    x_matrix = []
    y_array = []

    # populate X's and Y's with the given csv file
    csv_line = read_csv_file(csv_file)
    for row in csv_line:
        x_row, y = read_values_from_row(row)
        x_matrix.append(x_row)
        y_array.append(y)

    x_matrix = np.array(x_matrix)
    x_matrix = stats.zscore(x_matrix, axis=0)
    y_array = np.array(y_array)

    return x_matrix, y_array


def read_values_from_row(row=None):
    if row:
        x_row = []
        y = None

        for key, value in row.items():
            if key != 'image_class' and key != 'image_name':
                x_row.append(float(value))
            elif key == 'image_class':
                y = int(value)
            else:
                y = value

        return x_row, y
    else:
        return [], None


def read_csv_file(file_path=None):
    return csv.DictReader(open(file_path))
