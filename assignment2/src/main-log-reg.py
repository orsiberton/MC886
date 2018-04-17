import csv

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    x_matrix = []
    y_array = []

    # populate X's and Y's with the given csv file
    csv_line = read_csv_file('features.csv')
    for row in csv_line:
        x_row, y = read_values_from_row(row)
        x_matrix.append(x_row)
        y_array.append(y)

    x_matrix = np.array(x_matrix)
    y_array = np.array(y_array)

    # splits the train data into train and validation with validation being 20% of the original train data set
    x_train, x_validation, y_train, y_validation = train_test_split(x_matrix, y_array, test_size=0.20, random_state=0)

    # fit the LogReg
    classifier = LogisticRegression(solver='sag',
                                    max_iter=10000,
                                    multi_class='multinomial').fit(x_train, y_train)

    # validate the solution
    score = classifier.score(x_validation, y_validation)
    print("Training score : %.3f (%s)" % (score, 'multinomial'))

    predictions = classifier.predict(x_validation)
    plot_confusion_matrix(score, y_validation, predictions)


def plot_confusion_matrix(score, y_validation, predictions):
    cm = metrics.confusion_matrix(y_validation, predictions)
    print("Confusion matrix: ")
    print(cm)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    plt.show()


def read_values_from_row(row=None):
    if row:
        x_row = [
            float(row['gaussian_variance']),
            float(row['gaussian_kurtosis']),
            float(row['gaussian_skew']),
            float(row['denoised_variance']),
            float(row['denoised_kurtoses']),
            float(row['denoised_skew']),
        ]

        if 'image_class' in row.keys():
            y = int(row['image_class'])
        else:
            y = -1

        return x_row, y
    else:
        return [], -1


def read_csv_file(file_path=None):
    return csv.DictReader(open(file_path))


if __name__ == '__main__':
    main()
