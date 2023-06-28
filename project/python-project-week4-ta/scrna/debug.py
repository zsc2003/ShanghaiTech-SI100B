from sklearn.cluster import MiniBatchKMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
               [4, 2], [4, 0], [4, 4],
               [4, 5], [0, 1], [2, 2],
               [3, 2], [5, 5], [1, -1]])
# manually fit on batches
kmeans = MiniBatchKMeans(n_clusters=2,
                         random_state=0,
                         batch_size=6)
kmeans = kmeans.partial_fit(X[0:6,:])
kmeans = kmeans.partial_fit(X[6:12,:])

# fit on the whole data
kmeans = MiniBatchKMeans(n_clusters=2,
                         random_state=0,
                          batch_size=6,
                          max_iter=10).fit(X)
print(kmeans.cluster_centers_)
