import numpy as np
from assignment_2.utils.im2col import im2col, col2im
from .base import Layer

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, regularizer=None, regularizer_grad=None, reg_lambda=0.0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.regularizer = regularizer
        self.regularizer_grad = regularizer_grad
        self.reg_lambda = reg_lambda

        # Xavier Initialization
        scale = np.sqrt(1. / (in_channels * np.prod(self.kernel_size)))
        self.weights = np.random.randn(out_channels, in_channels, *self.kernel_size) * scale
        self.bias = np.zeros(out_channels)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        N, C, H, W = input_tensor.shape
        KH, KW = self.kernel_size

        self.out_h = (H + 2 * self.padding - KH) // self.stride + 1
        self.out_w = (W + 2 * self.padding - KW) // self.stride + 1

        self.cols = np.zeros((N, C * KH * KW, self.out_h * self.out_w))
        for i in range(N):
            self.cols[i], _, _ = im2col(input_tensor[i], KH, KW, self.stride, self.padding)

        self.col_W = self.weights.reshape(self.out_channels, -1)
        
        out = np.zeros((N, self.out_channels, self.out_h, self.out_w))
        for i in range(N):
            out[i] = (np.dot(self.col_W, self.cols[i]) + self.bias[:, np.newaxis]).reshape(self.out_channels, self.out_h, self.out_w)
        return out

    def backward(self, d_out, optimizer=None):
        N, _, _, _ = d_out.shape
        KH, KW = self.kernel_size

        dW_acc = np.zeros_like(self.weights)
        db_acc = np.zeros_like(self.bias)
        d_input_acc = np.zeros_like(self.input_tensor)

        for i in range(N):
            dout_sample = d_out[i]
            dout_flat = dout_sample.reshape(self.out_channels, -1)
            col_sample = self.cols[i]

            dW_sample = np.dot(dout_flat, col_sample.T).reshape(self.weights.shape)
            db_sample = np.sum(dout_flat, axis=1)
            dcol_sample = np.dot(self.col_W.T, dout_flat)
            
            # Use the original input shape for this sample for col2im
            d_input_sample = col2im(dcol_sample, self.input_tensor[i].shape, KH, KW, self.stride, self.padding)

            dW_acc += dW_sample
            db_acc += db_sample
            d_input_acc[i] = d_input_sample
        
        # Add regularization gradient if provided
        if self.regularizer_grad is not None and self.reg_lambda > 0.0:
            dW_acc += self.reg_lambda * self.regularizer_grad(self.weights)

        self.dW = dW_acc / N # Average gradients over the batch
        self.db = db_acc / N # Average gradients over the batch

        # Update weights and biases using optimizer if provided
        if optimizer is not None:
            self.weights = optimizer.update(self.weights, self.dW, 'conv_weights')
            self.bias = optimizer.update(self.bias, self.db, 'conv_bias')
        else:
            # Fallback to simple SGD if no optimizer provided
            learning_rate = 0.01
            self.weights -= learning_rate * self.dW
            self.bias -= learning_rate * self.db

        return d_input_acc

    def get_params(self):
        return {'weights': self.weights, 'bias': self.bias}
    
    def set_params(self, params):
        self.weights = params['weights']
        self.bias = params['bias']
