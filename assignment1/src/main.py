import csv


def main():
    csv_line = read_csv_file('../data/test.csv')

    for row in csv_line:
        print(row['url'], row['timedelta'])


def read_csv_file(file_path=None):
    return csv.DictReader(open(file_path))


if __name__ == "__main__":
    main()
