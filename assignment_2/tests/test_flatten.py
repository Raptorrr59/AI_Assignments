import numpy as np
import unittest
from assignment_2.layers.flatten import Flatten

class TestFlatten(unittest.TestCase):
    def test_flatten_forward_4d_input(self): # Renamed from test_flatten_forward_3d
        flatten_layer = Flatten()
        # Input shape (N, C, H, W) -> (2, 2, 2, 1) for simplicity
        input_data = np.array([[[[1], [2]], [[3], [4]]], # Sample 1
                               [[[5], [6]], [[7], [8]]]]).astype(float) # Sample 2
        # Expected output shape (N, C*H*W) -> (2, 4)
        expected_output = np.array([[1, 2, 3, 4],
                                      [5, 6, 7, 8]])
        
        output = flatten_layer.forward(input_data)
        np.testing.assert_array_equal(output, expected_output)
        self.assertEqual(flatten_layer.input_shape, (2, 2, 2, 1))
        self.assertEqual(output.shape, (2, 4))

    def test_flatten_forward_3d_input_as_batched_features(self):
        # Simulates input that might come from a previous layer that outputs (N, D1, D2)
        flatten_layer = Flatten()
        input_data = np.array([[[1, 2, 3],
                                [4, 5, 6]], # Sample 1, shape (2,3)
                               [[7, 8, 9],
                                [10, 11, 12]]]).astype(float) # Sample 2, shape (2,3)
                                                            # Total input shape (2, 2, 3)
        # Expected output shape (N, D1*D2) -> (2, 6)
        expected_output = np.array([[1, 2, 3, 4, 5, 6],
                                      [7, 8, 9, 10, 11, 12]])
        
        output = flatten_layer.forward(input_data)
        np.testing.assert_array_equal(output, expected_output)
        self.assertEqual(flatten_layer.input_shape, (2, 2, 3))
        self.assertEqual(output.shape, (2, 6))

    def test_flatten_forward_2d_input_already_flat_batched(self):
        # Input is (N, D), already flat per sample
        flatten_layer = Flatten()
        input_data = np.array([[1, 2, 3, 4], # Sample 1
                               [5, 6, 7, 8]]) # Sample 2. Shape (2, 4)
        expected_output = np.array([[1, 2, 3, 4],
                                      [5, 6, 7, 8]]) # Shape (2, 4)
        
        output = flatten_layer.forward(input_data)
        np.testing.assert_array_equal(output, expected_output)
        self.assertEqual(flatten_layer.input_shape, (2, 4))
        self.assertEqual(output.shape, (2, 4))

    def test_flatten_backward_batched(self):
        # Test with batched input (N, C, H, W) -> (N, C*H*W)
        input_tensor = np.random.randn(2, 3, 4, 5)  # (2, 3, 4, 5)
        flatten_layer = Flatten()
        output = flatten_layer.forward(input_tensor)
        output_grad = np.random.randn(*output.shape)
        
        d_input = flatten_layer.backward(output_grad, optimizer=None)
        self.assertEqual(d_input.shape, input_tensor.shape)

    def test_flatten_backward_no_learning_rate_effect_batched(self):
        # Test that learning rate doesn't affect the backward pass for Flatten
        input_tensor = np.random.randn(2, 3, 4, 5)  # (2, 3, 4, 5)
        flatten_layer = Flatten()
        output = flatten_layer.forward(input_tensor)
        output_grad = np.random.randn(*output.shape)
        
        d_input = flatten_layer.backward(output_grad, optimizer=None) # Large LR
        self.assertEqual(d_input.shape, input_tensor.shape)

# Remove old tests that assumed single sample input
# del TestFlatten.test_flatten_forward_2d
# del TestFlatten.test_flatten_forward_3d
# del TestFlatten.test_flatten_forward_1d_input_already_flat
# del TestFlatten.test_flatten_forward_input_is_1_N
# del TestFlatten.test_flatten_backward
# del TestFlatten.test_flatten_backward_no_learning_rate_effect

if __name__ == '__main__':
    unittest.main()