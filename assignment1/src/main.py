import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score

number_features = 51


def main():
    # number of iterations
    number_of_iterations = 50

    # definition of X's and Y's
    x_matrix = np.empty((0, number_features), dtype=float)
    y_array = np.array([])

    # learning rate
    alpha = 0.01

    # populate X's and Y's with the given csv file
    csv_line = read_csv_file('../data/train.csv')
    for row in csv_line:
        x_row, y = read_x_values_from_row(row)
        x_matrix = np.append(x_matrix, x_row, axis=0)
        y_array = np.append(y_array, y)

    print("Number of iterations: {}".format(number_of_iterations))
    print("Number of training examples: {}".format(x_matrix.shape[0]))
    print("Number of training features: {}".format(x_matrix.shape[1] - 1))
    print("Learning rate: {}".format(alpha))

    # learn from the data
    thetas, losses = gradient_descent(alpha, x_matrix, y_array, number_of_iterations)
    print("Thetas:\n {}".format(thetas))
    print("Losses:\n {}".format(losses))

    # show the cost over iterations
    plot_cost_function_x_iterations(number_of_iterations, losses)

    # apply the function with the new thetas
    validate_thetas(thetas)


def validate_thetas(thetas):
    # definition of X's and Y's for testing
    test_x_matrix = np.empty((0, number_features), dtype=float)
    test_y_array = np.array([])

    # read values for X's from the test csv
    test_x_csv_line = read_csv_file('../data/test.csv')
    for row in test_x_csv_line:
        x_row, y = read_x_values_from_row(row)
        test_x_matrix = np.append(test_x_matrix, x_row, axis=0)

    # read values for Y's from the test csv
    test_y_csv_line = read_csv_file('../data/test_target.csv')
    for row in test_y_csv_line:
        test_y_array = np.append(test_y_array, int(row['shares']))

    predictions = np.dot(test_x_matrix, thetas)
    # print(predictions.round())
    # print(test_y_array)
    print("Accuracy: {}".format(accuracy_score(test_y_array, predictions.round())))


def gradient_descent(alpha, x, y, number_of_interations):
    # number of training examples
    m = x.shape[0]

    # definition of the first thetas, the default value is one for all the thetas
    thetas = np.array(number_features * [1]).transpose()

    # transpose X's for gradient descent
    transposed_x = x.transpose()

    # array with all the losses over iterations
    losses = np.array([])

    for i in range(0, number_of_interations):
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


def read_x_values_from_row(row=None):
    if row:

        # TODO still have to deal with the DISCRETE values
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
            # float(row['kw_min_min']),
            # float(row['kw_max_min']),
            # float(row['kw_avg_min']),
            # float(row['kw_min_max']),
            # float(row['kw_max_max']),
            # float(row['kw_avg_max']),
            # float(row['kw_min_avg']),
            # float(row['kw_max_avg']),
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

        # normalize the features
        # norm_x_row = stats.norm.pdf(x_row)
        norm_x_row = stats.zscore(x_row, axis=1, ddof=1)

        # append the X0 value for theta 0
        norm_x_row = np.append([[1]], norm_x_row)
        norm_x_row = norm_x_row.reshape((1, number_features))

        if 'shares' in row.keys():
            y = int(row['shares'])
        else:
            y = 0

        return norm_x_row, y
    else:
        return np.empty((0, number_features)), 0


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

# print(x_matrix)

# matrix = np.empty((0, 7), dtype=float)
# print(matrix)
#
# array = np.array([[1, 2, 3, 4, 5, 6, 7]])
# print(array)
# matrix = np.append(matrix, array, axis=0)
# print(matrix)
#
# array = np.array([[1, 1, 1, 1, 1, 1, 1]])
# print(array)
# matrix = np.append(matrix, array, axis=0)
# print(matrix)
#
# array = np.array([[2, 2, 2, 2, 2, 2, 2]])
# print(array)
# matrix = np.append(matrix, array, axis=0)
#
# print(matrix)
# print(matrix[2])
#
# thetas = np.array([5, 5, 5, 5, 5, 5, 5]).transpose()
# print(thetas)
#
# y = np.array([1, 2, 5])
# print(np.append(y, 1))
#
# print(np.dot(matrix, thetas) - y)
# print(np.dot(matrix.transpose(), np.dot(matrix, thetas) - y) / 3)

# print(np.transpose(matrix[2]).dot(matrix[2]) - 10)
