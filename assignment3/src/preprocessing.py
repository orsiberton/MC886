import re
from string import punctuation

import pandas as pd


def main(year=2017, all=False):
    if all:
        df = pd.read_csv('../data/news_headlines_sample.csv')
    else:
        df = pd.read_csv('../data/news_headlines_{}.csv'.format(year))

    text_lower = [text.lower() for text in df.headline_text.values]
    text_letters = [''.join(c for c in headline if c not in punctuation) for headline in text_lower]

    text_final = [re.sub(r'[^A-Za-z]+', ' ', x) for x in text_letters]

    df.headline_text = text_final
    if all:
        df.to_csv('../data/news_headlines_sample_processed.csv')
    else:
        df.to_csv('../data/news_headlines_processed_{}.csv'.format(year))


if __name__ == '__main__':
    main()
