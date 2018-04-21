import csv

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn import metrics

import read_features

NUMBER_OF_FEATURES = 15 + 324

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

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y_train = np_utils.to_categorical(y_train)
    dummy_y_validation = np_utils.to_categorical(y_validation)

    model = Sequential()
    model.add(Dense(339, activation='relu', input_dim=NUMBER_OF_FEATURES))
    model.add(Dropout(rate=0.5))
    model.add(Dense(339, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(339, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(339, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(339, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(339, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, dummy_y_train, epochs=80, batch_size=64,
                        validation_data=(x_validation, dummy_y_validation), verbose=0)

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'bo')
    plt.plot(epochs, val_loss_values, 'b+')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()

    plt.clf()

    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.plot(epochs, acc_values, 'bo')
    plt.plot(epochs, val_acc_values, 'b+')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.show()

    print("[INFO] evaluating on testing set...")
    (loss, accuracy) = model.evaluate(x_validation, dummy_y_validation)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
                                                         accuracy * 100))

    predictions = model.predict(x_validation)

    predictions = [np.argmax(pred)for pred in predictions]

    cm = metrics.confusion_matrix(y_validation, predictions)
    print("Confusion matrix: ")
    print(cm)

    # class the images
    classify_test_images(model)


def classify_test_images(classifier):
    x_matrix, y_array = read_features.extract_features('features-False.csv')

    predictions = classifier.predict(x_matrix)

    with open('results.csv', 'w') as results_file:
        fieldnames = ["fname", "camera"]

        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, image_name in enumerate(y_array):
            writer.writerow({"fname": image_name, "camera": class_dict[str(np.argmax(predictions[index]))]})


if __name__ == '__main__':
    main()
