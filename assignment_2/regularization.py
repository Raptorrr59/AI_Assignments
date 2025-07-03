import numpy as np

def l1_penalty(weights):
    return np.sum(np.abs(weights))

def l1_gradient(weights):
    return np.sign(weights)

def l2_penalty(weights):
    return np.sum(weights ** 2)

def l2_gradient(weights):
    return 2 * weights

def elastic_net_penalty(weights, l1_ratio=0.5):
    return l1_ratio * l1_penalty(weights) + (1 - l1_ratio) * l2_penalty(weights)

def elastic_net_gradient(weights, l1_ratio=0.5):
    return l1_ratio * l1_gradient(weights) + (1 - l1_ratio) * l2_gradient(weights)
