import numpy as np
import unittest
from assignment_2.layers.conv2d import Conv2D
from assignment_2.utils.im2col import im2col, col2im

class TestConv2D(unittest.TestCase):
    def test_conv2d_initialization(self):
        conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.assertEqual(conv.weights.shape, (16, 3, 3, 3))
        self.assertEqual(conv.bias.shape, (16,))
        self.assertEqual(conv.stride, 1)
        self.assertEqual(conv.padding, 1)

    def test_conv2d_forward_simple(self):
        # Simple case: 1 channel input, 1 filter, no padding, stride 1
        conv = Conv2D(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
        # Manually set weights and bias for predictable output
        conv.weights = np.array([[[[1, 2], [3, 4]]]]).astype(float) # (1, 1, 2, 2)
        conv.bias = np.array([0.5]).astype(float)

        input_data = np.array([[[[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9]]]]).astype(float) # (1, 1, 3, 3)
        
        # Expected output calculation:
        # Region 1 (top-left): 1*1 + 2*2 + 4*3 + 5*4 = 1 + 4 + 12 + 20 = 37. Bias = 0.5. Output = 37.5
        # Region 2 (top-right): 2*1 + 3*2 + 5*3 + 6*4 = 2 + 6 + 15 + 24 = 47. Bias = 0.5. Output = 47.5
        # Region 3 (bottom-left): 4*1 + 5*2 + 7*3 + 8*4 = 4 + 10 + 21 + 32 = 67. Bias = 0.5. Output = 67.5
        # Region 4 (bottom-right): 5*1 + 6*2 + 8*3 + 9*4 = 5 + 12 + 24 + 36 = 77. Bias = 0.5. Output = 77.5
        expected_output = np.array([[[[37.5, 47.5],
                                      [67.5, 77.5]]]]) # Shape (1, 1, 2, 2)

        output = conv.forward(input_data) # Pass 4D input
        np.testing.assert_array_almost_equal(output, expected_output, decimal=5) # Compare with 4D expected
        self.assertEqual(output.shape, (1, 1, 2, 2)) # (N, out_channels, out_h, out_w)

    def test_conv2d_forward_padding_stride(self):
        conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
        conv.weights = np.ones((1, 1, 3, 3)).astype(float) # (1, 1, 3, 3)
        conv.bias = np.array([0.0]).astype(float)

        input_data = np.array([[[[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9]]]]).astype(float) # (1, 1, 3, 3)
        
        # With padding=1, input becomes:
        # 0 0 0 0 0
        # 0 1 2 3 0
        # 0 4 5 6 0
        # 0 7 8 9 0
        # 0 0 0 0 0
        # Kernel sum is 9.
        # Stride 2, kernel 3x3
        # Output H = (3 + 2*1 - 3)//2 + 1 = 2/2 + 1 = 2
        # Output W = (3 + 2*1 - 3)//2 + 1 = 2/2 + 1 = 2

        # Region 1 (top-left): (0*1+0*1+0*1 + 0*1+1*1+2*1 + 0*1+4*1+5*1) = 0+0+0+0+1+2+0+4+5 = 12
        # Region 2 (top-right): (0*1+0*1+0*1 + 2*1+3*1+0*1 + 5*1+6*1+0*1) = 0+0+0+2+3+0+5+6+0 = 16
        # Region 3 (bottom-left): (0*1+4*1+5*1 + 0*1+7*1+8*1 + 0*1+0*1+0*1) = 0+4+5+0+7+8+0+0+0 = 24
        # Region 4 (bottom-right): (5*1+6*1+0*1 + 8*1+9*1+0*1 + 0*1+0*1+0*1) = 5+6+0+8+9+0+0+0+0 = 28
        expected_output = np.array([[[[12., 16.],
                                      [24., 28.]]]]) # Shape (1, 1, 2, 2)
        output = conv.forward(input_data) # Pass 4D input
        np.testing.assert_array_almost_equal(output, expected_output, decimal=5) # Compare with 4D expected
        self.assertEqual(output.shape, (1, 1, 2, 2)) # (N, out_channels, out_h, out_w)

    def test_conv2d_backward_shape(self):
        conv = Conv2D(3, 2, 3)
        input_tensor = np.random.randn(1, 3, 5, 5)
        output = conv.forward(input_tensor)
        d_out = np.random.randn(*output.shape)
        
        d_input = conv.backward(d_out, optimizer=None)
        self.assertEqual(d_input.shape, input_tensor.shape)

    def test_conv2d_backward_gradient_check(self):
        # Use a very small network and input for numerical gradient checking
        N_batch = 1 # Batch size of 1 for gradient check
        in_channels = 1
        out_channels = 1
        kernel_size_int = 2 # kernel_size is an int here
        input_h, input_w = 3, 3

        conv = Conv2D(in_channels, out_channels, kernel_size_int, stride=1, padding=0)
        input_data_batch = np.random.randn(N_batch, in_channels, input_h, input_w)
        
        # Ensure weights and bias are not too large to avoid large gradients
        # Use *conv.kernel_size which is a tuple (kernel_size_int, kernel_size_int)
        conv.weights = np.random.randn(out_channels, in_channels, *conv.kernel_size) * 0.1 
        conv.bias = np.random.randn(out_channels) * 0.1

        original_weights = conv.weights.copy()
        original_bias = conv.bias.copy()
        
        # Calculate output dimensions for d_out_mock correctly
        expected_out_h = (input_h + 2 * conv.padding - conv.kernel_size[0]) // conv.stride + 1
        expected_out_w = (input_w + 2 * conv.padding - conv.kernel_size[1]) // conv.stride + 1
        d_out_mock = np.ones((N_batch, conv.out_channels, expected_out_h, expected_out_w))
        # dout_flat_mock = d_out_mock.reshape(conv.out_channels, -1) # Not directly used if analytical grads from conv.dW

        epsilon = 1e-5 # Standard epsilon for numerical gradient

        # --- Numerical gradient for weights ---
        numerical_dW = np.zeros_like(conv.weights)
        # Must reset weights and bias state before each numerical calculation for an element
        # This was done by resetting conv.weights to original_weights at the start of this block
        # and restoring individual elements. Let's ensure full reset for each perturbation group.

        temp_weights_for_num_grad = original_weights.copy()
        temp_bias_for_num_grad = original_bias.copy()

        for i in range(conv.weights.shape[0]):
            for j in range(conv.weights.shape[1]):
                for k_idx in range(conv.weights.shape[2]): 
                    for l_idx in range(conv.weights.shape[3]): 
                        # Perturb +epsilon
                        conv.weights = temp_weights_for_num_grad.copy()
                        conv.bias = temp_bias_for_num_grad.copy()
                        conv.weights[i, j, k_idx, l_idx] += epsilon
                        out_plus = conv.forward(input_data_batch)
                        loss_plus = np.sum(out_plus) # Sum over all elements in output batch

                        # Perturb -epsilon
                        conv.weights = temp_weights_for_num_grad.copy()
                        conv.bias = temp_bias_for_num_grad.copy()
                        conv.weights[i, j, k_idx, l_idx] -= epsilon
                        out_minus = conv.forward(input_data_batch)
                        loss_minus = np.sum(out_minus) # Sum over all elements in output batch
                        
                        numerical_dW[i, j, k_idx, l_idx] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Restore layer to original state for analytical gradient calculation
        conv.weights = original_weights.copy()
        conv.bias = original_bias.copy()
        
        # --- Analytical Gradients (from backward pass) ---
        _ = conv.forward(input_data_batch) # Perform forward pass to set internal states (e.g., self.cols)
        analytical_dInput = conv.backward(d_out_mock, optimizer=None) # Get grads without updates
        analytical_dW = conv.dW
        analytical_db = conv.db

        # --- Compare Numerical and Analytical Gradients ---
        np.testing.assert_array_almost_equal(numerical_dW, analytical_dW, decimal=3, # Reduced precision for stability
                                            err_msg="Weight gradients don't match")

        # --- Numerical gradient for bias ---
        numerical_db = np.zeros_like(conv.bias)
        temp_bias_for_num_grad_b = original_bias.copy()
        temp_weights_for_num_grad_b = original_weights.copy()

        for i in range(conv.bias.shape[0]):
            # Perturb +epsilon
            conv.weights = temp_weights_for_num_grad_b.copy()
            conv.bias = temp_bias_for_num_grad_b.copy()
            conv.bias[i] += epsilon
            out_plus = conv.forward(input_data_batch)
            loss_plus = np.sum(out_plus) # Sum over all elements in output batch

            # Perturb -epsilon
            conv.weights = temp_weights_for_num_grad_b.copy()
            conv.bias = temp_bias_for_num_grad_b.copy()
            conv.bias[i] -= epsilon
            out_minus = conv.forward(input_data_batch)
            loss_minus = np.sum(out_minus) # Sum over all elements in output batch

            numerical_db[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Restore layer to original state for comparison (already done for analytical_db)
        conv.weights = original_weights.copy()
        conv.bias = original_bias.copy()
        np.testing.assert_array_almost_equal(numerical_db, analytical_db, decimal=5,
                                            err_msg="Bias gradients don't match")

        # --- Numerical gradient for input ---
        numerical_dInput_batch = np.zeros_like(input_data_batch)
        # For dInput, we perturb input_data_batch, layer weights/bias remain original_weights/original_bias
        conv.weights = original_weights.copy()
        conv.bias = original_bias.copy()

        for n_b in range(input_data_batch.shape[0]): # Iterate over batch
            for c_idx in range(input_data_batch.shape[1]): # Iterate over channels
                for h_idx in range(input_data_batch.shape[2]): # Iterate over height
                    for w_idx in range(input_data_batch.shape[3]): # Iterate over width
                        input_data_plus = input_data_batch.copy()
                        input_data_plus[n_b, c_idx, h_idx, w_idx] += epsilon
                        # Use original weights/bias for these forward passes
                        conv.weights = original_weights.copy() 
                        conv.bias = original_bias.copy()
                        out_plus = conv.forward(input_data_plus)
                        loss_plus = np.sum(out_plus) # Sum over all elements in output batch

                        input_data_minus = input_data_batch.copy()
                        input_data_minus[n_b, c_idx, h_idx, w_idx] -= epsilon
                        conv.weights = original_weights.copy()
                        conv.bias = original_bias.copy()
                        out_minus = conv.forward(input_data_minus)
                        loss_minus = np.sum(out_minus) # Sum over all elements in output batch

                        numerical_dInput_batch[n_b, c_idx, h_idx, w_idx] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Restore layer to original state for comparison (already done for analytical_dInput)
        conv.weights = original_weights.copy()
        conv.bias = original_bias.copy()
        # The analytical_dInput is already calculated from the backward pass with d_out_mock
        np.testing.assert_array_almost_equal(numerical_dInput_batch, analytical_dInput, decimal=3, # Reduced precision
                                            err_msg="Input gradients don't match")

if __name__ == '__main__':
    unittest.main()