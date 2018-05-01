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
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

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
    tfidf_vectorizer = TfidfVectorizer(max_features=5000,
                                       use_idf=True,
                                       # ngram_range=(3, 4),
                                       stop_words='english',
                                       tokenizer=tokenize_and_stem)

    # Fit the vectorizer to text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(df.headline_text)
    terms = tfidf_vectorizer.get_feature_names()
    # print(terms)

    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(400) # arbitrary - indicated on SVD documentation
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    tfidf_matrix = lsa.fit_transform(tfidf_matrix)

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    # Kmeans++
    np.random.seed(42)
    x = []
    y = []

    for i in range(2, 20):
        # reduced_data = PCA(n_components=2).fit_transform(tfidf_matrix)
        # print(reduced_data)
        kmeans = KMeans(init='k-means++', n_clusters=i, n_init=4, random_state=3425, max_iter=300)
        kmeans.fit(tfidf_matrix)
        labels = kmeans.labels_
        clusters = labels.tolist()

        # clusters centers
        centroids = kmeans.cluster_centers_
        error = kmeans.inertia_
        # print(centroids)
        print(error)
        x.append(i)
        y.append(error)
        print "Solucao i ", i

    plt.scatter(x, y)
    plt.plot(x, y, '-o')
    plt.show()

    best_k = 20 #for now just example
    kmeans = KMeans(init='k-means++', n_clusters=best_k, n_init=4, random_state=3425, max_iter=300)
    kmeans.fit(tfidf_matrix)
    labels = kmeans.labels_
    clusters = labels.tolist()

    df = pd.DataFrame(dict(label=clusters, headline=df.headline_text))
    df.to_csv('../results/kmeans_clustered_DF.txt', sep=',', index=False)

if __name__ == '__main__':
    main()
