import csv
import random


def main():
    samples = []
    for i in range(2003, 2018):
        samples.append(get_sample(i))

    with open('../data/news_headlines_sample.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['publish_date', 'headline_text'])
        writer.writeheader()

        for i in range(0, len(samples)):
            for j in range(0, len(samples[i])):
                writer.writerow(samples[i][j])

    print("Samples generated!")


def get_sample(year=2017, number_of_samples=10000):
    print("Generating sample for year {}".format(year))

    csv_file = csv.DictReader(open('../data/news_headlines_{}.csv'.format(year)))
    news_per_year = []

    for csv_line in csv_file:
        publish_date, headline_text = read_values_from_row(csv_line)
        news_per_year.append({'publish_date': publish_date, 'headline_text': headline_text})

    sample_indexes = random.sample(range(1, len(news_per_year)), number_of_samples)
    result = []
    for index in sample_indexes:
        result.append(news_per_year[index])

    return result


def read_values_from_row(row=None):
    if row:
        return row['publish_date'], row['headline_text']
    else:
        return None, None


if __name__ == '__main__':
    main()
