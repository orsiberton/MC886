import csv

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import read_features

class_dict = {
    '0': 'Motorola-Droid-Maxx',
    '1': 'iPhone-4s',
    '2': 'LG-Nexus-5x',
    '3': 'Motorola-Nexus-6',
    '4': 'iPhone-6',
    '5': 'Sony-NEX-7',
    '6': 'Samsung-Galaxy-Note3',
    '7': 'HTC-1-M7',
    '8': 'Samsung-Galaxy-S4',
    '9': 'Motorola-X'
}


def main():
    x_matrix, y_array = read_features.extract_features('features-True.csv')

    print("Starting training...")

    # splits the train data into train and validation with validation being 20% of the original train data set
    x_train, x_validation, y_train, y_validation = train_test_split(x_matrix, y_array, test_size=0.20, random_state=0)

    # fit the LogReg
    classifier = LogisticRegression(solver='sag',
                                    max_iter=5000,
                                    multi_class='multinomial').fit(x_train, y_train)

    print("Training done!")

    # validate the solution
    score = classifier.score(x_validation, y_validation)
    print("Training score : %.3f (%s)" % (score, 'multinomial'))

    predictions = classifier.predict(x_validation)
    plot_confusion_matrix(score, y_validation, predictions)

    # class the images
    classify_test_images(classifier)


def classify_test_images(classifier):
    x_matrix, y_array = read_features.extract_features('features-False.csv')

    predictions = classifier.predict(x_matrix)

    with open('results.csv', 'w') as results_file:
        fieldnames = ["fname", "camera"]

        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, image_name in enumerate(y_array):
            writer.writerow({"fname": image_name, "camera": class_dict[str(predictions[index])]})


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


if __name__ == '__main__':
    main()
