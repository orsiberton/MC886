import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


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
    df.headline_text = df.headline_text.apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Using TFIDF vectorizer to convert convert words to Vector Space
    tfidf_vectorizer = TfidfVectorizer(max_features=20000,
                                       use_idf=True,
                                       stop_words='english',
                                       tokenizer=tokenize_and_stem)

    # Fit the vectorizer to text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(df.headline_text)

    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(50)  # arbitrary - indicated on SVD documentation
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    tfidf_matrix = lsa.fit_transform(tfidf_matrix)

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    # Kmeans++
    np.random.seed(42)
    x = []
    score = []
    y = []

    for i in range(2, 100, 1):
        kmeans = MiniBatchKMeans(init='k-means++', n_clusters=i, random_state=3425, max_iter=300)
        kmeans.fit(tfidf_matrix)

        error = kmeans.inertia_
        silhouette = metrics.silhouette_score(tfidf_matrix, kmeans.labels_, metric='euclidean', sample_size=1000)

        print("Cluster {}, error {}, silhouette {}".format(i, error, silhouette))
        score.append(silhouette)
        x.append(i)
        y.append(error)

    plt.scatter(x, y)
    plt.plot(x, y, '-o')
    plt.show()
    plt.close()

    plt.scatter(x, score)
    plt.plot(x, score, '-o')
    plt.show()

    best_k = 46  # for now just example
    kmeans = MiniBatchKMeans(init='k-means++', n_clusters=best_k, random_state=3425, max_iter=300)
    kmeans.fit(tfidf_matrix)
    labels = kmeans.labels_
    clusters = labels.tolist()

    df = pd.DataFrame(dict(label=clusters, headline=df.headline_text))
    df.to_csv('kmeans_clustered_DF.csv', sep=',', index=False)


if __name__ == '__main__':
    main()
