import unittest
import numpy as np
from assignment_2.layers.batchnorm import BatchNorm

class TestBatchNorm(unittest.TestCase):
    def setUp(self):
        self.num_features = 4
        self.bn = BatchNorm(self.num_features)

    def test_forward_training(self):
        self.bn.training = True
        x = np.random.randn(5, self.num_features)
        out = self.bn.forward(x)
        # Check output shape
        self.assertEqual(out.shape, x.shape)
        # Check mean and var close to 0, 1 (batch normed)
        np.testing.assert_allclose(np.mean(self.bn.normalized, axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(np.var(self.bn.normalized, axis=0), 1, atol=1e-4)

    def test_forward_inference(self):
        self.bn.training = False
        x = np.random.randn(5, self.num_features)
        # Simulate running stats
        self.bn.running_mean = np.random.randn(1, self.num_features)
        self.bn.running_var = np.abs(np.random.randn(1, self.num_features)) + 1e-2
        out = self.bn.forward(x)
        self.assertEqual(out.shape, x.shape)

    def test_backward(self):
        self.bn.training = True
        x = np.random.randn(5, self.num_features)
        out = self.bn.forward(x)
        grad_out = np.random.randn(*out.shape)
        grad_in = self.bn.backward(grad_out, optimizer=None)
        self.assertEqual(grad_in.shape, x.shape)

if __name__ == '__main__':
    unittest.main() 