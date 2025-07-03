from .base import Layer
import numpy as np

class Dropout(Layer):
    def __init__(self, dropout_rate=0.5):
        self.rate = dropout_rate
        self.mask = None
        self.training = True  # toggle manually during eval

    def forward(self, input):
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input.shape)
            return (input * self.mask) / (1 - self.rate)
        else:
            return input  # No dropout during inference

    def backward(self, output_grad, optimizer=None):
        return (output_grad * self.mask) / (1 - self.rate) if self.training else output_grad
