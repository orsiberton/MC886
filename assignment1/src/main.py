import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score

number_features = 59


def main():
    # number of iterations
    number_of_iterations = 5000

    # definition of X's and Y's
    x_matrix = np.empty((0, number_features - 1), dtype=float)
    y_array = np.array([])

    # learning rate
    alpha = 0.01

    # populate X's and Y's with the given csv file
    csv_line = read_csv_file('../data/train.csv')
    for row in csv_line:
        x_row, y = read_x_values_from_row(row)
        x_matrix = np.append(x_matrix, x_row, axis=0)
        y_array = np.append(y_array, y)

    x_matrix = stats.zscore(x_matrix, axis=0)
    x0 = np.array(x_matrix.shape[0] * [1]).transpose().reshape((x_matrix.shape[0], 1))
    x_matrix = np.append(x0, x_matrix, axis=1)

    print("Number of iterations: {}".format(number_of_iterations))
    print("Number of training examples: {}".format(x_matrix.shape[0]))
    print("Number of training features: {}".format(x_matrix.shape[1] - 1))
    print("Learning rate: {}".format(alpha))

    # learn from the data
    thetas, losses = gradient_descent(alpha, x_matrix, y_array, number_of_iterations)
    print("Thetas:\n {}".format(thetas))

    # show the cost over iterations
    plot_cost_function_x_iterations(number_of_iterations, losses)

    #thetas2 = normal_equation(x_matrix, y_array)

    # apply the function with the new thetas
    validate_thetas(thetas)


def validate_thetas(thetas):
    # definition of X's and Y's for testing
    test_x_matrix = np.empty((0, number_features - 1), dtype=float)
    test_y_array = np.array([])

    # read values for X's from the test csv
    test_x_csv_line = read_csv_file('../data/test.csv')
    for row in test_x_csv_line:
        x_row, y = read_x_values_from_row(row)
        test_x_matrix = np.append(test_x_matrix, x_row, axis=0)

    test_x_matrix = stats.zscore(test_x_matrix, axis=0)
    x0 = np.array(test_x_matrix.shape[0] * [1]).transpose().reshape((test_x_matrix.shape[0], 1))
    test_x_matrix = np.append(x0, test_x_matrix, axis=1)

    # read values for Y's from the test csv
    test_y_csv_line = read_csv_file('../data/test_target.csv')
    for row in test_y_csv_line:
        test_y_array = np.append(test_y_array, int(row['shares']))

    predictions = np.dot(test_x_matrix, thetas)

    error_histogram = np.array([0] * 10)
    print("Accuracy: {}".format(accuracy_score(test_y_array, predictions.round())))
    for i in range(0, predictions.size):
        print("Prediction: {} | Y: {} | %: {}".format(round(predictions[i]), test_y_array[i],
                                                      calculate_error_percentage(predictions[i], test_y_array[i])))
        position = int(np.trunc(calculate_error_percentage(predictions[i], test_y_array[i]) / 10))
        if position == 10:
            error_histogram[9] += 1
        else:
            error_histogram[position] += 1

    fig, ax = plt.subplots()
    ax.set_ylabel('# Predicts')
    ax.set_xlabel('accuracy in %')

    for i, error in enumerate(error_histogram):
        ax.bar(10 * i, error, 10, align='edge', color='blue')

    plt.show()


def calculate_error_percentage(predict, y):
    percentage = np.abs(np.round((predict * 100) / y))
    if percentage > 100.0:
        percentage = 100 - (percentage - 100)
        if percentage < 0.0:
            percentage = 0

    return percentage


def gradient_descent(alpha, x, y, number_of_iterations):
    # number of training examples
    m = x.shape[0]

    # definition of the first thetas, the default value is one for all the thetas
    # thetas = np.array(number_features * [random.randint(1000, 10000)]).transpose()
    thetas = np.transpose(np.random.rand(number_features) * 10)
    # thetas = np.random.rand(number_features) * 1000

    # transpose X's for gradient descent
    transposed_x = x.transpose()

    # array with all the losses over iterations
    losses = np.array([])

    for i in range(0, number_of_iterations):
        # Hypothesis is the evaluation of the function
        hypothesis = np.dot(x, thetas)

        # loss is the hypothesis minus y
        loss = hypothesis - y

        # calculate the cost function
        cost_j = np.sum(loss ** 2) / (2 * m)
        losses = np.append(losses, cost_j)

        # calculate the gradient descent
        gradient = np.dot(transposed_x, loss) / m

        # update all the thetas in the same time
        thetas = thetas - alpha * gradient

    return thetas, losses


def normal_equation(x, y):
    matrix = np.dot(x.transpose(), x)
    matrix = np.linalg.pinv(matrix)
    matrix = np.dot(matrix, x.transpose())
    matrix = np.dot(matrix, y)

    return matrix


def read_x_values_from_row(row=None):
    if row:
        x_row = np.array([[
            float(row['n_tokens_title']),
            float(row['n_tokens_content']),
            float(row['n_unique_tokens']),
            float(row['n_non_stop_words']),
            float(row['n_non_stop_unique_tokens']),
            float(row['num_hrefs']),
            float(row['num_self_hrefs']),
            float(row['num_imgs']),
            float(row['num_videos']),
            float(row['average_token_length']),
            float(row['num_keywords']),
            float(row['data_channel_is_lifestyle']),
            float(row['data_channel_is_entertainment']),
            float(row['data_channel_is_bus']),
            float(row['data_channel_is_socmed']),
            float(row['data_channel_is_tech']),
            float(row['data_channel_is_world']),
            float(row['kw_min_min']),
            float(row['kw_max_min']),
            float(row['kw_avg_min']),
            float(row['kw_min_max']),
            float(row['kw_max_max']),
            float(row['kw_avg_max']),
            float(row['kw_min_avg']),
            float(row['kw_max_avg']),
            float(row['kw_avg_avg']),
            float(row['self_reference_min_shares']),
            float(row['self_reference_max_shares']),
            float(row['self_reference_avg_sharess']),
            float(row['weekday_is_monday']),
            float(row['weekday_is_tuesday']),
            float(row['weekday_is_wednesday']),
            float(row['weekday_is_thursday']),
            float(row['weekday_is_friday']),
            float(row['weekday_is_saturday']),
            float(row['weekday_is_sunday']),
            float(row['is_weekend']),
            float(row['LDA_00']),
            float(row['LDA_01']),
            float(row['LDA_02']),
            float(row['LDA_03']),
            float(row['LDA_04']),
            float(row['global_subjectivity']),
            float(row['global_sentiment_polarity']),
            float(row['global_rate_positive_words']),
            float(row['global_rate_negative_words']),
            float(row['rate_positive_words']),
            float(row['rate_negative_words']),
            float(row['avg_positive_polarity']),
            float(row['min_positive_polarity']),
            float(row['max_positive_polarity']),
            float(row['avg_negative_polarity']),
            float(row['min_negative_polarity']),
            float(row['max_negative_polarity']),
            float(row['title_subjectivity']),
            float(row['title_sentiment_polarity']),
            float(row['abs_title_subjectivity']),
            float(row['abs_title_sentiment_polarity'])
        ]])

        if 'shares' in row.keys():
            y = int(row['shares'])
        else:
            y = 0

        return x_row, y
    else:
        return np.empty((0, number_features - 1)), 0


def plot_cost_function_x_iterations(number_of_iterations, losses):
    fig, ax = plt.subplots()
    ax.set_ylabel('Loss')
    ax.set_xlabel('# iterations')
    fig.suptitle("Training Loss")

    plt.plot(np.arange(0, number_of_iterations), losses)
    plt.show()


def read_csv_file(file_path=None):
    return csv.DictReader(open(file_path))


if __name__ == "__main__":
    main()
