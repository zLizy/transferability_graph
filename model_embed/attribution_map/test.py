import os
import numpy as np
import pandas as pd

def _print(name,value,level=2):
    print()
    if level == 1:
        print('=====================')
    elif level > 1:
        print('---------------')
    print(f'== {name}: {value}')

def main():
    empty_any = []
    empty_all = []
    non_empty = []

    INPUT_SHAPE = 224
    explain_methods = ['input_x_gradient']#'elrp','saliency',
    explain_method = explain_methods[0]

    root = './feature'
    datasets = os.listdir(root)

    count = 0
    for dataset in datasets:
        _print('dataset',dataset,1)
        for file in os.listdir(os.path.join(root,dataset)):
            if explain_method in file:
                path = os.path.join(root,dataset,file)
                tmp = np.load(path)
                # size = len(tmp.tolist())
                if np.isnan(tmp).all():
                    empty_all.append(path)
                    print(f'{path} is empty')
                elif np.isnan(tmp).any():
                    empty_all.append(path)
                    print(f'{path} is empty')
                else:
                    non_empty.append(path)
                    count += 1
    print(non_empty)
    print(count)

if __name__ == '__main__':
    main()