import numpy as np

class ReLU:
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_grad, learning_rate):
        return output_grad * (self.input > 0)

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, input):
        self.input = input
        return np.where(input > 0, input, self.alpha * input)

    def backward(self, output_grad, learning_rate):
        return output_grad * np.where(self.input > 0, 1, self.alpha)
