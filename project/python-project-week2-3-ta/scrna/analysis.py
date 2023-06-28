from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from numpy.lib.function_base import interp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from typing import Optional, Tuple
import pandas as pd

def pca(data: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """    
    - Dimensionality Reduction Method: Principal component analysis (PCA).
    - Uses the implementation of *scikit-learn*.
    - The expected output DataFrame has n_components columns whose names are: PC0, PC1, ...
    """
    pca = PCA(n_components)
    pca.fit(data,n_components)
    PCs = pd.DataFrame(pca.transform(data), index=data.index, columns=["PC0", "PC1"])
    return PCs

def tsne(data: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """    
    - Dimensionality Reduction Method: T-distributed Stochastic Neighbor Embedding (TSNE).
    - Uses the implementation of *scikit-learn*.
    - The expected output DataFrame has n_components columns whose names are: PC0, PC1, ...
    """
    tsne = TSNE(n_components, init='random')#, learning_rate='auto' 加上这个会UFuncTypeError!!!!!!!!!!!!!!!!!
    PCs = pd.DataFrame(tsne.fit_transform(data), index=data.index, columns=["PC0", "PC1"])
    return PCs

def visualize(data: pd.DataFrame, label: Optional[pd.DataFrame] = None) -> None:
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
    data = pd.concat([data, pd.DataFrame(l, index=data.index, columns=["label"])], axis=1, sort=False)
    data.plot.scatter(x="PC0", y="PC1", c="label", colormap='Paired')
    pass

def cluster(data: pd.DataFrame, k: Optional[int] = 8, label: Optional[pd.DataFrame] = None) -> pd.DataFrame:
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
    """

    import numpy as np
    #print(data)
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
    statistics = pd.concat([statistics, pd.DataFrame(l, index=statistics.index, columns=["label"])], axis=1, sort=False)
    group = statistics.groupby(["label"]).mean()
    #print(statistics)
    group=np.array(group)
    #print(group)
    #init_pos=np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [0.8, 0.8]], np.float64)
    kmeans = KMeans(n_clusters=k, init=group, n_init=1).fit(data)
    # kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    labels = pd.DataFrame(kmeans.labels_, index=data.index, columns=['label'])
    return labels

def evaluate(data: pd.DataFrame, ground_truth: pd.DataFrame) -> Tuple[float, float]:
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
    """
    # print(data)
    # print(ground_truth)
    index = data.index
    new_label = list(data.loc[:,'label'])
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
    #print(new_label)
    #print(origin_label)
    NMI = normalized_mutual_info_score(new_label, origin_label)
    ARI = adjusted_rand_score(new_label, origin_label)
    return (NMI, ARI)