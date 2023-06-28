from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from numpy.lib.function_base import interp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MeanShift, AffinityPropagation, \
    MiniBatchKMeans
from sklearn.manifold import TSNE
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from functools import wraps
import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def timer_decorator(func):
    """
    An NB decorator.

    :param func: function
    :return:ret
    """
    @wraps(func)
    def wrapped_function(*args, **kwargs):
        print(f'{func.__name__} starts...')
        x = datetime.datetime.now()
        ret = func(*args, **kwargs)
        y = datetime.datetime.now()
        print(f'{func.__name__} running time:  {y - x}s')
        print('-----------------------')
        return ret
    return wrapped_function


@timer_decorator
def log_normalize(data: pd.DataFrame) -> pd.DataFrame:
    sums = data.sum(axis=1)
    data = data * 2500
    data = data.div(sums, axis=0)
    data = np.log1p(data)
    return data


@timer_decorator
def pca(data: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    - Dimensionality Reduction Method: Principal component analysis (PCA).
    - Uses the implementation of *scikit-learn*.
    - The expected output DataFrame has n_components columns whose names are: PC0, PC1, ...
    """
    data = log_normalize(data)
    # n_components = min(len(data.index), len(data.columns))

    pca = PCA(n_components=n_components, whiten=True)
    X_pca = pca.fit_transform(X=data)
    # print(pca.explained_variance_ratio_)
    PCs = pd.DataFrame(X_pca,
                       index=data.index,
                       columns=[f'PC{i}' for i in range(len(X_pca[0]))]
                       )
    return PCs


@timer_decorator
def tsne(data: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """    
    - Dimensionality Reduction Method: T-distributed Stochastic Neighbor Embedding (TSNE).
    - Uses the implementation of *scikit-learn*.
    - The expected output DataFrame has n_components columns whose names are: PC0, PC1, ...
    """
    tsne = TSNE(n_components=n_components, init='random', learning_rate=200.0)
    X_tsne = tsne.fit_transform(X=data)
    PCs = pd.DataFrame(X_tsne,
                       index=data.index,
                       columns=[f'PC{i}' for i in range(len(X_tsne[0]))]
                       )
    return PCs


@timer_decorator
def visualize(data: pd.DataFrame,
              label: Optional[pd.DataFrame] = None,
              algorithm: str = 'unsupervised K-means') -> None:
    """
    - Draw scatter plot in embedding coordinates (e.g. PCA, T-SNE).
    - Note that the color of the same label should be same, and vice versa.

    --------------------
    Parameters
    --------------------
    data
        The dataframe whose rows correspond to cells and columns to two embeddings.
    label
        Optional labels for coloring the scatter plot.
    --------------------
    :param algorithm:
    """
    a = data.index
    l = []
    tot = -1
    d = {}
    for i in a:
        if type(label.index[0]) == int:
            s = label.loc[int(i), "label"]
        else:
            s = label.loc[i, "label"]
        if s in d:
            l.append(d[s])
        else:
            tot += 1
            d[s] = tot
            l.append(tot)

    assert algorithm == 'unsupervised K-means' or 'supervised LDA', 'invalid algorithm name'
    if algorithm == 'unsupervised K-means':
        data = pd.concat([data, pd.DataFrame(l, index=data.index, columns=["label"])], axis=1, sort=False)
    elif algorithm == 'supervised LDA':
        test_size = int(0.25 * data.shape[0])
        data = data[-test_size:]
        l = l[-test_size:]
        data = pd.concat([data, pd.DataFrame(l, index=data.index, columns=["label"])], axis=1, sort=False)

    data.plot.scatter(x="PC0", y="PC1", c="label", colormap='Paired')
    pass


@timer_decorator
def cluster(data: pd.DataFrame,
            k: Optional[int] = 8,
            label: Optional[pd.DataFrame] = None,
            algorithm: str = 'unsupervised K-means') -> pd.DataFrame:
    """
    - Cluster cells into subgroups using Kmeans.
    - Uses the implementation of *scikit-learn*.
    - The expected output DataFrame has one column named 'label'.

    --------------------
    Parameters
    --------------------
    data
        The dataframe whose rows correspond to cells and columns to two embeddings.
    k
        The number of subgroups.
    --------------------
    :param algorithm:
    :param k:
    :param data:
    :param label:
    """
    statistics = data
    a = statistics.index
    l = []
    tot = -1
    d = {}
    for i in a:
        if type(label.index[0]) == int:
            s = label.loc[int(i), "label"]
        else:
            s = label.loc[i, "label"]
        if s in d:
            l.append(d[s])
        else:
            tot += 1
            d[s] = tot
            l.append(tot)
    label_index = l
    statistics = pd.concat([statistics, pd.DataFrame(label_index, index=statistics.index, columns=["label"])],
                           axis=1, sort=False)

    group = statistics.groupby(["label"])
    group_mean = np.array(group.mean())

    assert algorithm == 'unsupervised K-means' or 'supervised LDA', 'invalid algorithm name'
    if algorithm == 'unsupervised K-means':
        model = KMeans(n_clusters=k, init=group_mean, n_init=1, algorithm='auto')  # no improvements from elkan/full

        # model = AgglomerativeClustering(n_clusters=k)    # not much differences
        # model = SpectralClustering(n_clusters=k, assign_labels='discretize') # highly volatile; discretize is better?
        # model = MeanShift()   # no accuracy
        # model = AffinityPropagation()  # a mixture
        # model = MiniBatchKMeans(n_clusters=k, init=group_mean, n_init=1)    # maybe better?

        model.fit(X=data)
        y = model.labels_
        labels = pd.DataFrame(y, index=data.index, columns=['label'])

    elif algorithm == 'supervised LDA':
        # X_train, X_test, y_train, y_test = train_test_split(data, label_index, test_size=0.25)
        test_size = int(0.25 * data.shape[0])
        X_train = data.iloc[:-test_size]
        X_test = data.iloc[-test_size:]
        y_train = label_index[:-test_size]
        y_test = label_index[-test_size:]

        model = Pipeline([('LDA', LinearDiscriminantAnalysis()),
                          ])  # nothing useful, just to try and get familiar with the usage of Pipeline
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y = y_train + list(y_pred)

        labels = pd.DataFrame(y, index=data.index, columns=['label'])
        # print(model.score(X_test, y_test))

    else:
        labels = None

    return labels


@timer_decorator
def evaluate(data: pd.DataFrame,
             ground_truth: pd.DataFrame,
             algorithm: str = 'unsupervised K-means')-> Tuple[float, float]:
    """
    - Evaluate the clustering performance using metrics Normalized Mutual Information (NMI) 
      and Adjusted Rand Index (ARI).
    - Uses the implementation of *scikit-learn*.

    --------------------
    Parameters
    --------------------
    data
        The labels generated according to the clustering result.
    ground_truth
        The ground truth label.
    --------------------

    --------------------
    Returns
    --------------------
    A tuple (NMI, ARI)
    --------------------
    :param algorithm:
    """
    NMI, ARI = None, None
    # print(data)
    # print(ground_truth)
    index = data.index
    new_label = list(data.loc[:, 'label'])
    origin_label = []

    d = {}
    tot = -1
    for i in index:
        if type(ground_truth.index[0]) == int:
            s = ground_truth.loc[int(i), "label"]
        else:
            s = ground_truth.loc[i, "label"]
        if s in d:
            origin_label.append(d[s])
        else:
            tot += 1
            d[s] = tot
            origin_label.append(tot)
    # print(new_label)
    # print(origin_label)

    assert algorithm == 'unsupervised K-means' or 'supervised LDA', 'invalid algorithm name'
    if algorithm == 'unsupervised K-means':
        NMI = normalized_mutual_info_score(new_label, origin_label)
        ARI = adjusted_rand_score(new_label, origin_label)

    elif algorithm == 'supervised LDA':
        test_size = int(0.25 * data.shape[0])
        y_pred = new_label[-test_size:]
        y_test = origin_label[-test_size:]

        NMI = normalized_mutual_info_score(y_pred, y_test)
        ARI = adjusted_rand_score(y_pred, y_test)

    return NMI, ARI