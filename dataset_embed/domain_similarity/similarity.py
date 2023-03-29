import numpy as np
import time
import pyemd
from sklearn.metrics.pairwise import euclidean_distances
import os

gamma = 0.01
DIM = 2048
MAX_NUM_SAMPLES = 5000
# min_num_imgs

domains = ['food101', 'cifar10', 'cifar100']
LEN = len(domains)

feature_dir = './feature/resnet50/'
for dataset in domains:
    # Load extracted features on CUB-200.
    feature = np.load(feature_dir + dataset + f'_feature_{MAX_NUM_SAMPLES}.npy')
    label = np.load(feature_dir + dataset + f'_label_{MAX_NUM_SAMPLES}.npy')
    # Calculate class feature as the averaged features among all images of the class.
    # Class weight is defined as the number of images of the class.
    sorted_label = sorted(list(set(label)))
    feature_per_class = np.zeros((len(sorted_label), DIM), dtype=np.float32)
    weight = np.zeros((len(sorted_label), ), dtype=np.float32)
    counter = 0
    for i in sorted_label:
        idx = [(l==i) for l in label]
        feature_per_class[counter, :] = np.mean(feature[idx, :], axis=0)
        weight[counter] = np.sum(idx)
        counter += 1

    print('Feature per class shape: (%d, %d)' % (feature_per_class.shape[0], 
                                                feature_per_class.shape[1]))

    np.save(feature_dir + dataset + '.npy', feature_per_class)
    np.save(feature_dir + dataset + '_weight.npy', weight)

tic = time.time()
for i in range(LEN):
    for j in range(i+1,LEN):
        sd = domains[i]
        td = domains[j]
        print('%s --> %s' % (sd, td))
        f_s = np.load(feature_dir + sd + f'.npy')
        f_t = np.load(feature_dir + td + f'.npy')
        w_s = np.load(feature_dir + sd + f'_weight.npy')
        w_t = np.load(feature_dir + td + f'_weight.npy')

        # Remove source domain classes with number of images < 'min_num_imgs'.
        # idx = [i for i in range(len(w_s)) if w_s[i] >= min_num_imgs]
        # f_s = f_s[idx, :]
        # w_s = w_s[idx]

        # Make sure two histograms have the same length and distance matrix is square.
        data = np.float64(np.append(f_s, f_t, axis=0))
        w_1 = np.zeros((len(w_s) + len(w_t),), np.float64)
        w_2 = np.zeros((len(w_s) + len(w_t),), np.float64)
        w_1[:len(w_s)] = w_s / np.sum(w_s)
        w_2[len(w_s):] = w_t / np.sum(w_t)
        D = euclidean_distances(data, data)

        emd = pyemd.emd(np.float64(w_1), np.float64(w_2), np.float64(D))
        print('EMD: %.3f    Domain Similarity: %.3f\n' % (emd, np.exp(-gamma*emd)))
print('Elapsed time: %.3fs' % (time.time() - tic))