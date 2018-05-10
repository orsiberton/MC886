import csv


def main():
    csv_file = csv.DictReader(open('../data/news_headlines.csv'))

    news_per_year = {}
    for i in range(2003, 2018):
        news_per_year[i] = []

    for csv_line in csv_file:
        publish_date, headline_text = read_values_from_row(csv_line)
        year = int(publish_date[:4])

        news_per_year[year].append({'publish_date': publish_date, 'headline_text': headline_text})

    for i in range(2003, 2018):
        with open('../data/news_headlines_{}.csv'.format(i), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['publish_date', 'headline_text'])
            writer.writeheader()

            for j in range(0, len(news_per_year[i])):
                writer.writerow(news_per_year[i][j])


def read_values_from_row(row=None):
    if row:
        return row['publish_date'], row['headline_text']
    else:
        return None, None


if __name__ == '__main__':
    main()
