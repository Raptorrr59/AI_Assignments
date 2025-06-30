import numpy as np

class FullyConnected:
    def __init__(self, input_size, output_size):
        # Xavier initialization
        # Weights: (input_size, output_size) for easier dot product with (N, input_size)
        scale = np.sqrt(1. / input_size) 
        self.weights = np.random.randn(input_size, output_size) * scale
        self.bias = np.zeros((1, output_size)) # Bias: (1, output_size) for broadcasting

    def forward(self, input_tensor):
        # input_tensor shape: (N, input_size)
        self.input_tensor = input_tensor
        # Output shape: (N, input_size) @ (input_size, output_size) -> (N, output_size)
        return np.dot(input_tensor, self.weights) + self.bias

    def backward(self, output_grad, learning_rate):
        # output_grad shape: (N, output_size)
        N = self.input_tensor.shape[0]

        # dW shape: (input_size, N) @ (N, output_size) -> (input_size, output_size)
        dW = np.dot(self.input_tensor.T, output_grad)
        # db shape: sum over N -> (1, output_size)
        db = np.sum(output_grad, axis=0, keepdims=True)
        # d_input shape: (N, output_size) @ (output_size, input_size) -> (N, input_size)
        d_input = np.dot(output_grad, self.weights.T)

        # Update parameters
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

        return d_input
