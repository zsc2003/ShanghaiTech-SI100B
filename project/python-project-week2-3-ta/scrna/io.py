from scipy.io import mmread
import pandas as pd


def generate_df(matrix_path: str, gene_path: str, barcode_path: str) -> pd.DataFrame:
    """
    - Read count matrix, gene, barcode from given file and merge to a dataframe.
    - You need to set indices, column labels and their names. For every genes, we only need to use gene symbols as
    column labels.
    - You may need to read the documentation on docs.scipy.org for more information. Here is a reference:
    https://docs.scipy.org/doc/scipy/reference/tutorial/io.html#matrix-market-files
    """
    print(barcode_path)
    with open(barcode_path, 'r') as b:
        barcodes = b.readlines()
    with open(gene_path, 'r') as g:
        genes = g.readlines()
    matrix = mmread(matrix_path)

    # processing the raw attrs
    barcodes = list(map(lambda x: x.strip('\n'), barcodes))
    genes = list(map(lambda x: x.split('\t')[1].strip('\n'), genes))

    df = pd.DataFrame(matrix.todense().T, index=barcodes, columns=genes, dtype=int)
    df.columns.name = 'genes'
    df.index.name = 'barcodes'

    return df
