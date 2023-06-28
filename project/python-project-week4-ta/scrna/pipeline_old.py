def whole_pipeline(matrix_path: str, gene_path: str, barcode_path: str):
    """
    - Automatize your whole pipeline, that you can do io, processing, 
      analysising in one run.
      
    - Meanwhile, please reorganize your code and do some refactroing
      in order to make it cleaner to read and easier to maintain.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import scipy

    from scrna.io import generate_df
    df = generate_df(matrix_path, gene_path, barcode_path)

    from scrna.preprocess import n_genes_by_counts
    qc_n_genes_by_counts = n_genes_by_counts(df)
    plt.figure(1)
    plt.hist(qc_n_genes_by_counts, bins=1000)
    plt.xlabel('N genes by counts')
    plt.ylabel('N cells')

    from scrna.preprocess import total_counts
    qc_total_counts = total_counts(df)
    plt.figure(2)
    plt.hist(qc_total_counts, bins=1000)
    plt.xlabel('Total counts')
    plt.ylabel('N cells')

    from scrna.preprocess import highest_expr_genes
    qc_highest_expr_genes = highest_expr_genes(df, 5)
    ground_truth = pd.read_csv("data/testdata/ground_truth.csv")
    df = df[:ground_truth.shape[0]]
    from scrna.preprocess import filter_cells
    from scrna.preprocess import filter_genes
    filtered_df = filter_genes(df, min_cells=3)
    filtered_df = filter_cells(filtered_df, min_genes=500)
    filtered_df = filter_cells(filtered_df, max_genes=1000)

    from scrna.analysis import pca
    pca_result = pca(filtered_df)

    from scrna.analysis import tsne
    tsne_result = tsne(filtered_df)

    
    from scrna.analysis import visualize
    from scrna.analysis import cluster
    plt.figure(3)
    visualize(pca_result, label=ground_truth)
    plt.title("pca")

    pca_cluster_result = cluster(pca_result, 8, label=ground_truth)
    plt.figure(4)
    visualize(pca_result, label=pca_cluster_result)
    plt.title("pca_cluster")

    plt.figure(5)
    visualize(tsne_result, label=ground_truth)
    plt.title("tsne")

    tsne_cluster_result = cluster(tsne_result, 8, label = ground_truth)
    plt.figure(6)
    visualize(tsne_result, label=tsne_cluster_result)
    plt.title("tsne_cluster")

    from scrna.analysis import evaluate

    pca_nmi, pca_ari = evaluate(pca_cluster_result, ground_truth)
    tsne_nmi, tsne_ari = evaluate(tsne_cluster_result, ground_truth)
    print(pca_nmi, pca_ari)
    print(tsne_nmi, tsne_ari)
    return