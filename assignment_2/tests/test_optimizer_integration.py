import unittest
import numpy as np
from assignment_2.layers.fullyconnected import FullyConnected
from assignment_2.layers.batchnorm import BatchNorm
from assignment_2.optimizers import SGD, Adam

class TestOptimizerIntegration(unittest.TestCase):
    def test_fc_with_sgd_optimizer(self):
        """Test that FullyConnected layer works with SGD optimizer"""
        fc = FullyConnected(3, 2)
        sgd = SGD(lr=0.01)
        
        # Store initial weights
        initial_weights = fc.weights.copy()
        initial_bias = fc.bias.copy()
        
        # Forward pass
        x = np.random.randn(2, 3)
        output = fc.forward(x)
        
        # Backward pass with optimizer
        grad = np.random.randn(*output.shape)
        fc.backward(grad, optimizer=sgd)
        
        # Check that weights were updated (should be different from initial)
        self.assertFalse(np.allclose(fc.weights, initial_weights))
        self.assertFalse(np.allclose(fc.bias, initial_bias))

    def test_fc_with_adam_optimizer(self):
        """Test that FullyConnected layer works with Adam optimizer"""
        fc = FullyConnected(3, 2)
        adam = Adam(lr=0.001)
        
        # Store initial weights
        initial_weights = fc.weights.copy()
        initial_bias = fc.bias.copy()
        
        # Forward pass
        x = np.random.randn(2, 3)
        output = fc.forward(x)
        
        # Backward pass with optimizer
        grad = np.random.randn(*output.shape)
        fc.backward(grad, optimizer=adam)
        
        # Check that weights were updated (should be different from initial)
        self.assertFalse(np.allclose(fc.weights, initial_weights))
        self.assertFalse(np.allclose(fc.bias, initial_bias))

    def test_batchnorm_with_optimizer(self):
        """Test that BatchNorm layer works with optimizer"""
        bn = BatchNorm(4)
        sgd = SGD(lr=0.01)
        
        # Store initial parameters
        initial_gamma = bn.gamma.copy()
        initial_beta = bn.beta.copy()
        
        # Forward pass
        x = np.random.randn(5, 4)
        output = bn.forward(x)
        
        # Backward pass with optimizer
        grad = np.random.randn(*output.shape)
        bn.backward(grad, optimizer=sgd)
        
        # Check that parameters were updated
        self.assertFalse(np.allclose(bn.gamma, initial_gamma))
        self.assertFalse(np.allclose(bn.beta, initial_beta))

    def test_fc_fallback_to_sgd(self):
        """Test that FullyConnected falls back to SGD when no optimizer provided"""
        fc = FullyConnected(3, 2)
        
        # Store initial weights
        initial_weights = fc.weights.copy()
        initial_bias = fc.bias.copy()
        
        # Forward pass
        x = np.random.randn(2, 3)
        output = fc.forward(x)
        
        # Backward pass without optimizer (should use fallback)
        grad = np.random.randn(*output.shape)
        fc.backward(grad, optimizer=None)
        
        # Check that weights were updated
        self.assertFalse(np.allclose(fc.weights, initial_weights))
        self.assertFalse(np.allclose(fc.bias, initial_bias))

if __name__ == '__main__':
    unittest.main() 