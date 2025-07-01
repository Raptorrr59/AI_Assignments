# optimizers/sgd.py
import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def update(self, param, grad, name):
        if self.momentum > 0:
            if name not in self.velocity or self.velocity[name].shape != grad.shape:
                self.velocity[name] = np.zeros_like(grad)
            self.velocity[name] = self.momentum * self.velocity[name] - self.lr * grad
            return param + self.velocity[name]
        else:
            return param - self.lr * grad
