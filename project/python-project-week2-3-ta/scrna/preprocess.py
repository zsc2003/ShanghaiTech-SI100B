import pandas as pd
import numpy as np
from typing import Optional, List

def n_genes_by_counts(data: pd.DataFrame) -> np.ndarray:
    '''
    - Calculate the number of genes with at least 1 count in a cell for all cells.
    - The result ndarray should be flat with shape (2700, ).
    - For example ret[i] is the number of genes with at least 1 count in the i-th cell.
    '''
    a=[]
    for i in range(2700):
        x=data.iloc[i]>0
        a.append(x.sum())
    ret=np.array(a)
    return ret

def total_counts(data: pd.DataFrame) -> np.ndarray:
    '''
    - Calculate the total number of counts in a cell for all cells.
    - The result ndarray should be flat with shape (2700, ).
    - For example ret[i] is the number of counts of the i-th cell.
    '''
    a=[]
    for i in range(2700):
        x=data.iloc[i].sum()
        a.append(x)
    ret=np.array(a)
    return ret

def highest_expr_genes(data: pd.DataFrame, ntop: int = 10) -> List[str]:
    '''
    - Calculate, for each cell, the fraction of counts assigned to every genes within
    that cell and output the `n_top` genes with the highest mean fraction over all cells.
    '''
    title=data.columns
    d = {}
    for i in range(32738):
        x=data.iloc[:,i].sum()
        if title[i] in d:
            d[title[i]]=max(d[title[i]],x)
        else:
            d[title[i]]=x
    #print(d)
    d=sorted(d.items(), key=lambda x: x[1], reverse=1)
    ret=[]
    for i in range(ntop):
        ret.append(d[i][0])
    #print(ret)
    return ret

def filter_cells(data: pd.DataFrame,
min_counts: Optional[int] = None,
min_genes: Optional[int] = None,
max_counts: Optional[int] = None,
max_genes: Optional[int] = None
) -> pd.DataFrame:
    '''
    - Filter cell outliers based on counts and numbers of genes expressed.

    - For instance, only keep cells with at least `min_counts` counts or
    `min_genes` genes expressed. This is to filter measurement outliers,
    i.e. “unreliable” observations.

    - It is guaranteed that only one of the optional parameters `min_counts`, `min_genes`,
    `max_counts`, `max_genes` will be provided per call.

    Parameters
    ----------
    data
        The dataframe whose rows correspond to cells and columns to genes.
    min_counts
        Minimum number of counts required for a cell to pass filtering.
    min_genes
        Minimum number of genes expressed required for a cell to pass filtering.
    max_counts
        Maximum number of counts required for a cell to pass filtering.
    max_genes
        Maximum number of genes expressed required for a cell to pass filtering.
    '''
    n_given_options = sum(option is not None for option in [min_genes, min_counts, max_genes, max_counts])
    if n_given_options != 1:
        raise ValueError(
            'Only provide one of the optional parameters `min_counts`, '
            '`min_genes`, `max_counts`, `max_genes` per call.'
        )

    a=[]
    if min_counts is not None:
        for i in range(data.shape[0]):
            x = data.iloc[i]
            if x.sum() < min_counts:
                a.append(i)

    if max_counts is not None:
        for i in range(data.shape[0]):
            x = data.iloc[i]
            if x.sum() > max_counts:
                a.append(i)

    if min_genes is not None:
        for i in range(data.shape[0]):
            x = data.iloc[i] > 0
            if x.sum() < min_genes:
                a.append(i)
    if max_genes is not None:
        for i in range(data.shape[0]):
            x = data.iloc[i] > 0
            if x.sum() > max_genes:
                a.append(i)
    ret = data.drop(data.index[a])
    return ret



def filter_genes(data: pd.DataFrame,
min_counts: Optional[int] = None,
min_cells: Optional[int] = None,
max_counts: Optional[int] = None,
max_cells: Optional[int] = None
) -> pd.DataFrame:
    '''
    - Filter genes based on number of cells or counts.

    - Keep genes that have at least `min_counts` counts or are expressed in at
    least `min_cells` cells or have at most `max_counts` counts or are expressed
    in at most `max_cells` cells.

    - It is guaranteed that only one of the optional parameters `min_counts`, `min_cells`,
    `max_counts`, `max_cells` per call will be provided.


    Parameters
    ----------
    data
        The dataframe whose rows correspond to cells and columns to genes.
    min_counts
        Minimum number of counts required for a gene to pass filtering.
    min_cells
        Minimum number of cells expressed required for a gene to pass filtering.
    max_counts
        Maximum number of counts required for a gene to pass filtering.
    max_cells
        Maximum number of cells expressed required for a gene to pass filtering.
    '''
    n_given_options = sum(option is not None for option in [min_cells, min_counts, max_cells, max_counts])
    if n_given_options != 1:
        raise ValueError(
            'Only provide one of the optional parameters `min_counts`, '
            '`min_cells`, `max_counts`, `max_cells` per call.'
        )
    a = []
    if min_counts is not None:
        for i in range(data.shape[1]):
            x = data.iloc[:, i]
            if x.sum() >= min_counts:
                a.append(i)

    if max_counts is not None:
        for i in range(data.shape[1]):
            x = data.iloc[:, i]
            if x.sum() <= max_counts:
                a.append(i)

    if min_cells is not None:
        for i in range(data.shape[1]):
            x = data.iloc[:, i] > 0
            if x.sum() >= min_cells:
                a.append(i)

    if max_cells is not None:
        for i in range(data.shape[1]):
            x = data.iloc[:, i] > 0
            if x.sum() <= max_cells:
                a.append(i)

    ret = data.iloc[:, a]
    return ret

# TODO: maybe add a bonus task here, highly_variable_genes
