import numpy as np

class Flatten:
    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape  # Save for backward (N, C, H, W) or (N, D)
        N = input_tensor.shape[0]
        # Reshape to (N, -1) to flatten all dimensions except batch
        return input_tensor.reshape(N, -1)

    def backward(self, output_grad, optimizer=None):
        # output_grad has shape (N, flattened_dims)
        # Reshape it back to the original input_shape (N, C, H, W) or (N, D)
        return output_grad.reshape(self.input_shape)
