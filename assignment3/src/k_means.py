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


def main(year=2017, all=False, svd=False):
    if all:
        df = pd.read_csv('../data/news_headlines_sample_processed.csv')
    else:
        df = pd.read_csv('../data/news_headlines_processed_{}.csv'.format(year))

    # text data in dataframe and removing stops words
    stop_words = set(stopwords.words('english'))
    df.headline_text = df.headline_text.apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Using TFIDF vectorizer to convert convert words to Vector Space
    tfidf_vectorizer = TfidfVectorizer(max_features=500,
                                       use_idf=True,
                                       # analyzer='char',
                                       ngram_range=(1, 2),
                                       # min_df=0.1,
                                       stop_words='english',
                                       tokenizer=tokenize_and_stem)
    print tfidf_vectorizer

    # Fit the vectorizer to text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(df.headline_text)
    # print tfidf_vectorizer.vocabulary_
    idf = tfidf_vectorizer.idf_
    word_dict =  dict(zip(tfidf_vectorizer.get_feature_names(), idf))
    # print word_dict['australia']

    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    if svd:
        svd = TruncatedSVD(50)  # arbitrary - indicated on SVD documentation
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        tfidf_matrix = lsa.fit_transform(tfidf_matrix)

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))
    else:
        print "help"

    # Kmeans++
    np.random.seed(42)
    x = []
    score = []
    y = []

    for i in range(30, 150, 1):
        kmeans = MiniBatchKMeans(init='k-means++', n_clusters=i, random_state=3425, max_iter=300)
        kmeans.fit_predict(tfidf_matrix)

        error = kmeans.inertia_
        silhouette = metrics.silhouette_score(tfidf_matrix, kmeans.labels_, metric='euclidean', sample_size=1000)

        # print("Cluster {}, error {}, silhouette {}".format(i, error, silhouette))
        score.append(silhouette)
        x.append(i)
        y.append(error)

    plt.title('Inertia x K')
    plt.scatter(x, y)
    plt.plot(x, y, '-o')
    if all:
        plt.savefig('Inertia_word_gram_1_50_features.png')
    else:
        plt.savefig('Inertia_word_gram_1_50_features_{}.png'.format(year))
    plt.show()
    plt.close()

    plt.title('Average silhouette score x K')
    plt.scatter(x, score)
    plt.ylim(-0.2, 1)
    plt.plot(x, score, '-o')
    if all:
        plt.savefig('Silhouette_word_gram_1_50_features.png')
    else:
        plt.savefig('Silhouette_word_gram_1_50_features_{}.png'.format(year))
    plt.show()


    best_k = 80  # for now just example
    kmeans = MiniBatchKMeans(init='k-means++', n_clusters=best_k, random_state=3425, max_iter=300)
    kmeans.fit_predict(tfidf_matrix)
    labels = kmeans.labels_
    clusters = labels.tolist()

    terms_per_doc = tfidf_vectorizer.inverse_transform(tfidf_matrix)
    feature_phrases = map(lambda x: ' '.join(x), terms_per_doc)

    df = pd.DataFrame(dict(label=clusters, headline=feature_phrases))
    if all:
        df.to_csv('kmeans_clustered_DF.csv', sep=',', index=False)
    else:
        df.to_csv('kmeans_clustered_DF_{}.csv'.format(year), sep=',', index=False)


if __name__ == '__main__':
    main()
