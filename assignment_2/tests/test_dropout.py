import unittest
import numpy as np
from assignment_2.layers.dropout import Dropout

class TestDropout(unittest.TestCase):
    def setUp(self):
        self.dropout = Dropout(dropout_rate=0.5)

    def test_forward_training(self):
        self.dropout.training = True
        x = np.ones((10, 10))
        out = self.dropout.forward(x)
        # Output should have zeros and scaled non-zeros
        self.assertEqual(out.shape, x.shape)
        # Check that some values are zero (dropped)
        self.assertTrue(np.any(out == 0))
        # Check scaling: nonzero values should be 1/(1-rate)
        nonzero = out[out != 0]
        self.assertTrue(np.allclose(nonzero, 1/(1-self.dropout.rate)))

    def test_forward_inference(self):
        self.dropout.training = False
        x = np.ones((10, 10))
        out = self.dropout.forward(x)
        # No dropout during inference
        np.testing.assert_array_equal(out, x)

    def test_backward_training(self):
        self.dropout.training = True
        x = np.ones((10, 10))
        out = self.dropout.forward(x)
        grad_out = np.ones_like(x)
        grad_in = self.dropout.backward(grad_out, optimizer=None)
        # Backward should also apply mask and scaling
        self.assertEqual(grad_in.shape, x.shape)
        self.assertTrue(np.any(grad_in == 0))
        nonzero = grad_in[grad_in != 0]
        self.assertTrue(np.allclose(nonzero, 1/(1-self.dropout.rate)))

    def test_backward_inference(self):
        self.dropout.training = False
        x = np.ones((10, 10))
        self.dropout.forward(x)
        grad_out = np.ones_like(x)
        grad_in = self.dropout.backward(grad_out, optimizer=None)
        np.testing.assert_array_equal(grad_in, grad_out)

if __name__ == '__main__':
    unittest.main() 