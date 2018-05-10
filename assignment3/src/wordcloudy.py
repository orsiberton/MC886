import pandas as pd
import numpy as np
from wordcloud import WordCloud

def main():
    df = pd.read_csv('kmeans_clustered_DF.csv')
    clusters = df.label.unique()

    for cluster in clusters:
        headlines = df.values[np.where(df.values[:,1] == cluster)]
        text = '\t'.join([l[0] for l in headlines])
        # Generate a word cloud image
        wordcloud = WordCloud().generate(text)

        # Display the generated image:
        # the matplotlib way:
        import matplotlib.pyplot as plt
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


if __name__ == '__main__':
    main()
