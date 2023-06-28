from functools import wraps
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy


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
        return ret
    return wrapped_function


@timer_decorator   # merely display the running time, does not matter
def whole_pipeline(matrix_path: str, gene_path: str, barcode_path: str, pca_n: int):
    """
    - Automatize your whole pipeline, that you can do io, processing, 
      analyzing in one run.
      
    - Meanwhile, please reorganize your code and do some refactroing
      in order to make it cleaner to read and easier to maintain.
    """

    # 2.1.1 read the data
    from scrna.io import generate_df
    df = generate_df(matrix_path, gene_path, barcode_path)

    # 2.2.1 n genes by counts
    from scrna.preprocess import n_genes_by_counts
    qc_n_genes_by_counts = n_genes_by_counts(df)
    plt.figure(1)
    plt.hist(qc_n_genes_by_counts, bins=1000)
    plt.xlabel('N genes by counts')
    plt.ylabel('N cells')

    # 2.2.2 total counts
    from scrna.preprocess import total_counts
    qc_total_counts = total_counts(df)
    plt.figure(2)
    plt.hist(qc_total_counts, bins=1000)
    plt.xlabel('Total counts')
    plt.ylabel('N cells')

    # 2.2.3 highest expr genes
    from scrna.preprocess import highest_expr_genes
    qc_highest_expr_genes = highest_expr_genes(df, 5)
    ground_truth = pd.read_csv("data/testdata/ground_truth.csv")
    df = df[:ground_truth.shape[0]]
    print(f'shape of data: {df.shape}')

    # 2.2.4 filter cells
    # 2.2.5 filter genes
    from scrna.preprocess import filter_cells
    from scrna.preprocess import filter_genes
    filtered_df = filter_genes(df, min_cells=3)
    filtered_df = filter_cells(filtered_df, min_genes=500)
    filtered_df = filter_cells(filtered_df, max_genes=1000)
    print(f'shape of filtered data: {filtered_df.shape}')

    # 3.1.1 dimensionality reduction
    # PCA
    from scrna.analysis import pca
    pca_result = pca(filtered_df, n_components=pca_n)

    # t-SNE
    from scrna.analysis import tsne
    tsne_result = tsne(filtered_df)

    # 3.2.1 visualization
    from scrna.analysis import visualize
    plt.figure(3)
    visualize(pca_result, label=ground_truth)
    plt.title("pca")
    plt.figure(5)
    visualize(tsne_result, label=ground_truth)
    plt.title("tsne")

    # 3.3.1 perform clustering
    from scrna.analysis import cluster
    pca_cluster_result = cluster(pca_result, 8, label=ground_truth)
    plt.figure(4)
    visualize(pca_result, label=pca_cluster_result)
    plt.title("pca_cluster")
    tsne_cluster_result = cluster(tsne_result, 8, label=ground_truth)
    plt.figure(6)
    visualize(tsne_result, label=tsne_cluster_result)
    plt.title("tsne_cluster")

    # 3.3.2 evaluate the clustering performance
    from scrna.analysis import evaluate
    pca_nmi, pca_ari = evaluate(pca_cluster_result, ground_truth)
    tsne_nmi, tsne_ari = evaluate(tsne_cluster_result, ground_truth)
    print(f'pca_nmi:  {pca_nmi}   pca_ari:  {pca_ari}')
    print(f'tsne_nmi: {tsne_nmi}  tsne_ari: {tsne_ari}')
    return

