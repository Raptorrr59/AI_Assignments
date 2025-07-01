# optimizers/adam.py
import numpy as np
from .base import Optimizer

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = {}

    def update(self, param, grad, name):
        if name not in self.m or self.m[name].shape != grad.shape:
            self.m[name] = np.zeros_like(grad)
            self.v[name] = np.zeros_like(grad)
            self.t[name] = 0

        self.t[name] += 1
        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[name] / (1 - self.beta1 ** self.t[name])
        v_hat = self.v[name] / (1 - self.beta2 ** self.t[name])

        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
