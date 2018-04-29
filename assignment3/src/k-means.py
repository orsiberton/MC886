from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import os
from sklearn.decomposition import PCA
import numpy as np

# Tokenizer to return stemmed words, we use this
def tokenize_and_stem(text_file):
    # declaring stemmer and stopwords language
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text_file)
    filtered = [w for w in words if w not in stop_words]
    stems = [stemmer.stem(t) for t in filtered]
    return stems

def main():
    df = pd.read_csv('../data/news_headlines_processed.csv')

    # text data in dataframe and removing stops words
    stop_words = set(stopwords.words('english'))
    df.headline_text = df.headline_text.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Using TFIDF vectorizer to convert convert words to Vector Space
    tfidf_vectorizer = TfidfVectorizer(max_features=200000,
                                       use_idf=True,
                                       ngram_range=(3, 4),
                                       stop_words='english',
                                       tokenizer=tokenize_and_stem)

    # Fit the vectorizer to text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(df.headline_text)
    terms = tfidf_vectorizer.get_feature_names()
    # print(terms)

    # Kmeans++
    np.random.seed(42)
    x = []
    y = []

    for i in range(20, 40):
        # reduced_data = PCA(n_components=2).fit_transform(tfidf_matrix)
        # print(reduced_data)
        kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10, random_state=3425)
        kmeans.fit(tfidf_matrix)

        # clusters centers
        centroids = kmeans.cluster_centers_
        error = kmeans.inertia_
        # print(centroids)
        print(error)
        x.append(i)
        y.append(error)

    plt.scatter(x, y)
    plt.plot(x, y, '-o')
    plt.show()

    # km = KMeans(n_clusters=7, init='k-means++', max_iter=300, n_init=1, verbose=0, random_state=3425)
    # km.fit(tfidf_matrix)
    # labels = km.labels_
    # clusters = labels.tolist()
    # print clusters
    # # Calculating the distance measure derived from cosine similarity
    # distance = 1 - cosine_similarity(tfidf_matrix)

if __name__ == '__main__':
    main()
