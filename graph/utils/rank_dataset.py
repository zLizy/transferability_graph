import pandas as pd
import numpy as np
import os

def rank_by_dataset(test_dataset,root='..'):
    path = f'{root}/doc/corr_domain_similarity_{test_dataset}.csv'
    df_corr = pd.read_csv(path,index_col=0)
    corr = df_corr[test_dataset].sort_values().iloc[:15]
    print(corr)
    pass

if __name__ == '__main__':
    rank_by_dataset('cifar100','../..')