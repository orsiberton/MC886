print(__doc__)

import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

# #############################################################################
# Visualize the results on PCA-reduced data

for i in range(1, 15):
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
    kmeans.fit(reduced_data)

    # clusters centers
    centroids = kmeans.cluster_centers_
    error = kmeans.inertia_
    # print(centroids)
    print(error)
