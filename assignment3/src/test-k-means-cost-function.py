import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

x = []
y = []

for i in range(1, 20):
    reduced_data = PCA(n_components=2).fit_transform(data)
    # print(reduced_data)
    kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
    kmeans.fit(reduced_data)

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
