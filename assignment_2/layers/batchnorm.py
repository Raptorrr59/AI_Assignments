import numpy as np
from .base import Layer

class BatchNorm(Layer):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Running stats for inference
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

    def get_params(self):
        return {
            'gamma': self.gamma,
            'beta': self.beta,
            'running_mean': self.running_mean,
            'running_var': self.running_var
        }

    def set_params(self, params):
        self.gamma = params.get('gamma', self.gamma)
        self.beta = params.get('beta', self.beta)
        self.running_mean = params.get('running_mean', self.running_mean)
        self.running_var = params.get('running_var', self.running_var)

    def forward(self, input):
        self.input = input  # shape: (1, features)

        if self.training:
            self.batch_mean = np.mean(input, axis=0, keepdims=True)
            self.batch_var = np.var(input, axis=0, keepdims=True)
            self.normalized = (input - self.batch_mean) / np.sqrt(self.batch_var + self.eps)

            # Update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            # Use running statistics during inference
            self.normalized = (input - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * self.normalized + self.beta

    def backward(self, output_grad, optimizer=None):
        N = self.input.shape[0]
        std_inv = 1. / np.sqrt(self.batch_var + self.eps)

        dx_norm = output_grad * self.gamma
        dvar = np.sum(dx_norm * (self.input - self.batch_mean) * -0.5 * std_inv**3, axis=0, keepdims=True)
        dmean = np.sum(dx_norm * -std_inv, axis=0, keepdims=True) + dvar * np.mean(-2 * (self.input - self.batch_mean), axis=0, keepdims=True)

        dx = dx_norm * std_inv + dvar * 2 * (self.input - self.batch_mean) / N + dmean / N
        dgamma = np.sum(output_grad * self.normalized, axis=0, keepdims=True)
        dbeta = np.sum(output_grad, axis=0, keepdims=True)

        # Update learnable parameters using optimizer if provided
        if optimizer is not None:
            self.gamma = optimizer.update(self.gamma, dgamma, 'bn_gamma')
            self.beta = optimizer.update(self.beta, dbeta, 'bn_beta')
        else:
            # Fallback to simple SGD if no optimizer provided
            learning_rate = 0.01
            self.gamma -= learning_rate * dgamma
            self.beta -= learning_rate * dbeta

        return dx
