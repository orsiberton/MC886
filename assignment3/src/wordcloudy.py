import pandas as pd
import numpy as np
from wordcloud import WordCloud

def main(year=2017, all=False):
    if all:
        df = pd.read_csv('kmeans_clustered_DF.csv')
    else:
        df = pd.read_csv('kmeans_clustered_DF_{}.csv'.format(year))
    clusters = df.label.unique()

    for cluster in clusters:
        headlines = df.values[np.where(df.values[:,1] == cluster)]
        text = '\t'.join([l[0] for l in headlines])
        # Generate a word cloud image
        wordcloud = WordCloud(collocations=False).generate(text)

        # Display the generated image:
        # the matplotlib way:
        import matplotlib.pyplot as plt
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


if __name__ == '__main__':
    main()
