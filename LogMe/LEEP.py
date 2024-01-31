import numpy as np
from numpy import exp

def LEEP(pseudo_source_label: np.ndarray, target_label: np.ndarray):
    """

    :param pseudo_source_label: shape [N, C_s]
    :param target_label: shape [N], elements in [0, C_t)
    :return: leep score
    """
    # print(f'label:{target_label[:2]}')
    # print(f'prediction:{pseudo_source_label[:1]}')
    N, C_s = pseudo_source_label.shape
    target_label = target_label.reshape(-1)
    C_t = int(np.max(target_label) + 1)   # the number of target classes
    normalized_prob = pseudo_source_label / float(N)  # sum(normalized_prob) = 1
    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for joint distribution over (y, z)
    for i in range(C_t):
        this_class = normalized_prob[target_label == i]
        row = np.sum(this_class, axis=0)
        joint[i] = row
    p_target_given_source = (joint / joint.sum(axis=0, keepdims=True)).T  # P(y | z)

    empirical_prediction = pseudo_source_label @ p_target_given_source
    # print(f'empirical_prediction: {empirical_prediction[:5]}')
    empirical_prob = np.array([softmax(predict)[int(label)] for predict, label in zip(empirical_prediction, target_label)])
    # print(f'empirical_prob: {min(empirical_prob)}')
    leep_score = np.mean(np.log(empirical_prob))
    return leep_score

 
# calculate the softmax of a vector
def softmax(vector):
 e = exp(vector)
 return e / e.sum()