import numpy as np
import unittest
from assignment_2.layers.relu import ReLU, LeakyReLU

class TestReLU(unittest.TestCase):
    def test_relu_forward(self):
        relu = ReLU()
        # Batched input (N, D1, D2) -> (2, 2, 3)
        input_data = np.array([[[-1, 0, 1],
                                [-2, 2, -3]],
                               [[-0.5, 0.5, 0],
                                [1.5, -1.5, 2.5]]])
        expected_output = np.array([[[0, 0, 1],
                                   [0, 2, 0]],
                                  [[0, 0.5, 0],
                                   [1.5, 0, 2.5]]])
        output = relu.forward(input_data)
        np.testing.assert_array_equal(output, expected_output)
        self.assertEqual(output.shape, input_data.shape) # Ensure shape is preserved

    def test_relu_backward(self):
        relu = ReLU()
        input_data = np.array([[[-1, 0, 1],
                                [-2, 2, -3]],
                               [[-0.5, 0.5, 0],
                                [1.5, -1.5, 2.5]]])
        relu.forward(input_data)  # Store input for backward pass
        output_grad = np.ones_like(input_data) # Gradient of 1 for all elements
        expected_d_input = np.array([[[0, 0, 1],
                                    [0, 1, 0]],
                                   [[0, 1, 0],
                                    [1, 0, 1]]])
        d_input = relu.backward(output_grad, learning_rate=0.01)
        np.testing.assert_array_equal(d_input, expected_d_input)
        self.assertEqual(d_input.shape, input_data.shape)

    def test_relu_backward_with_gradient(self):
        relu = ReLU()
        input_data = np.array([[[-0.5, 0.5], [-1.5, 1.5]],
                               [[2.0, -2.0], [0.0, 3.0]]]) # Shape (2,2,2)
        relu.forward(input_data)
        output_grad = np.array([[[0.1, 0.2], [0.3, 0.4]],
                                [[0.5, 0.6], [0.7, 0.8]]])
        expected_d_input = np.array([[[0, 0.2], [0, 0.4]],
                                     [[0.5, 0], [0, 0.8]]])
        d_input = relu.backward(output_grad, learning_rate=0.01)
        np.testing.assert_array_almost_equal(d_input, expected_d_input, decimal=5)
        self.assertEqual(d_input.shape, input_data.shape)

class TestLeakyReLU(unittest.TestCase):
    def test_leaky_relu_initialization(self):
        leaky_relu = LeakyReLU(alpha=0.05)
        self.assertEqual(leaky_relu.alpha, 0.05)

        leaky_relu_default = LeakyReLU()
        self.assertEqual(leaky_relu_default.alpha, 0.01) # Default alpha

    def test_leaky_relu_forward(self):
        alpha = 0.1
        leaky_relu = LeakyReLU(alpha=alpha)
        input_data = np.array([[[-1, 0, 1], [-2, 2, 3]],
                               [[-0.5, 0.5, 1.0], [1.5, -1.5, 2.0]]]) # (2,2,3)
        expected_output = np.where(input_data > 0, input_data, alpha * input_data)
        output = leaky_relu.forward(input_data)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=5)
        self.assertEqual(output.shape, input_data.shape)

    def test_leaky_relu_backward(self):
        alpha = 0.1
        leaky_relu = LeakyReLU(alpha=alpha)
        input_data = np.array([[[-1, 0, 1], [-2, 2, 3]],
                               [[-0.5, 0.5, 1.0], [1.5, -1.5, 2.0]]]) # (2,2,3)
        leaky_relu.forward(input_data)  # Store input for backward pass
        output_grad = np.ones_like(input_data)
        expected_d_input = np.where(input_data > 0, 1, alpha)
        d_input = leaky_relu.backward(output_grad, learning_rate=0.01)
        np.testing.assert_array_almost_equal(d_input, expected_d_input, decimal=5)
        self.assertEqual(d_input.shape, input_data.shape)

    def test_leaky_relu_backward_with_gradient(self):
        alpha = 0.05
        leaky_relu = LeakyReLU(alpha=alpha)
        input_data = np.array([[[-0.5, 0.5], [-1.5, 1.5]],
                               [[2.0, -2.0], [0.0, 3.0]]]) # Shape (2,2,2)
        leaky_relu.forward(input_data)
        output_grad = np.array([[[0.1, 0.2], [0.3, 0.4]],
                                [[0.5, 0.6], [0.7, 0.8]]])
        expected_d_input = np.array([[[0.1*alpha, 0.2*1], [0.3*alpha, 0.4*1]],
                                     [[0.5*1, 0.6*alpha], [0.7*alpha, 0.8*1]]])
        d_input = leaky_relu.backward(output_grad, learning_rate=0.01)
        np.testing.assert_array_almost_equal(d_input, expected_d_input, decimal=5)
        self.assertEqual(d_input.shape, input_data.shape)

if __name__ == '__main__':
    unittest.main()