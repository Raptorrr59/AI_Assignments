import numpy as np
import unittest
from assignment_2.layers.fullyconnected import FullyConnected

class TestFullyConnected(unittest.TestCase):
    def test_fc_initialization(self):
        fc = FullyConnected(input_size=10, output_size=5)
        self.assertEqual(fc.weights.shape, (10, 5)) # (input_size, output_size)
        self.assertEqual(fc.bias.shape, (1, 5))    # (1, output_size)
        # Check Xavier initialization (scale part)
        # Not checking random values, but scale should be applied
        self.assertTrue(np.all(fc.weights != 0)) # Assuming random won't be all zeros
        self.assertTrue(np.all(fc.bias == 0))

    def test_fc_forward_batched(self): # Renamed and updated
        fc = FullyConnected(input_size=3, output_size=2)
        # Manually set weights and bias for predictable output
        # Weights: (input_size, output_size) -> (3, 2)
        fc.weights = np.array([[1, 4],
                               [2, 5],
                               [3, 6]]).astype(float)
        # Bias: (1, output_size) -> (1, 2)
        fc.bias = np.array([[0.5, -0.5]]).astype(float)

        # Input data: (N, input_size) -> (2, 3)
        input_data = np.array([[0.1, 0.2, 0.3],  # Sample 1
                               [0.4, 0.5, 0.6]]) # Sample 2
        
        # Expected output calculation (N, output_size) -> (2, 2)
        # Sample 1:
        # Out1_1 = (0.1*1 + 0.2*2 + 0.3*3) + 0.5 = (0.1 + 0.4 + 0.9) + 0.5 = 1.4 + 0.5 = 1.9
        # Out1_2 = (0.1*4 + 0.2*5 + 0.3*6) - 0.5 = (0.4 + 1.0 + 1.8) - 0.5 = 3.2 - 0.5 = 2.7
        # Sample 2:
        # Out2_1 = (0.4*1 + 0.5*2 + 0.6*3) + 0.5 = (0.4 + 1.0 + 1.8) + 0.5 = 3.2 + 0.5 = 3.7
        # Out2_2 = (0.4*4 + 0.5*5 + 0.6*6) - 0.5 = (1.6 + 2.5 + 3.6) - 0.5 = 7.7 - 0.5 = 7.2
        expected_output = np.array([[1.9, 2.7],
                                    [3.7, 7.2]])

        output = fc.forward(input_data)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=5)
        self.assertEqual(output.shape, (2, 2))
        np.testing.assert_array_equal(fc.input_tensor, input_data) # Check input is stored

    def test_fc_backward_shape_batched(self): # Renamed and updated
        N = 4
        input_size = 5
        output_size = 3
        fc = FullyConnected(input_size=input_size, output_size=output_size)
        input_data = np.random.rand(N, input_size)
        _ = fc.forward(input_data)
        
        output_grad = np.random.rand(N, output_size) # Gradient from next layer
        d_input = fc.backward(output_grad, learning_rate=0.01)
        
        self.assertEqual(d_input.shape, (N, input_size)) # Matches forward input shape
        self.assertEqual(fc.weights.shape, (input_size, output_size)) # Check weights shape after update
        self.assertEqual(fc.bias.shape, (1, output_size))    # Check bias shape after update

    def test_fc_backward_gradient_values_and_updates_batched(self): # Renamed and updated
        N = 2
        input_size = 3
        output_size = 2
        learning_rate = 0.1
        fc = FullyConnected(input_size=input_size, output_size=output_size)
        
        # Weights (input_size, output_size) -> (3,2)
        fc.weights = np.array([[0.1, 0.4],
                               [0.2, 0.5],
                               [0.3, 0.6]]).astype(float)
        # Bias (1, output_size) -> (1,2)
        fc.bias = np.array([[0.1, 0.2]]).astype(float)
        initial_weights = fc.weights.copy()
        initial_bias = fc.bias.copy()

        # Input (N, input_size) -> (2,3)
        input_data = np.array([[1.0, 2.0, 3.0],  # Sample 1
                               [4.0, 5.0, 6.0]]) # Sample 2
        _ = fc.forward(input_data)

        # Output grad (N, output_size) -> (2,2)
        output_grad = np.array([[0.5, 1.0],   # Grad for sample 1
                                [0.2, 0.4]])  # Grad for sample 2

        # Calculate expected gradients and updates manually
        # dW = input_tensor.T @ output_grad / N
        # input_tensor.T (3,2) @ output_grad (2,2) -> (3,2)
        # For dW[0,0]: (1*0.5 + 4*0.2) / 2 = (0.5 + 0.8)/2 = 1.3/2 = 0.65
        # For dW[0,1]: (1*1.0 + 4*0.4) / 2 = (1.0 + 1.6)/2 = 2.6/2 = 1.3
        # ... and so on
        expected_dW = np.dot(input_data.T, output_grad) / N
        
        # db = sum(output_grad, axis=0, keepdims=True) / N
        # db_sum_axis0 = [0.5+0.2, 1.0+0.4] = [0.7, 1.4]
        # expected_db = [[0.7/2, 1.4/2]] = [[0.35, 0.7]]
        expected_db = np.sum(output_grad, axis=0, keepdims=True) / N
        
        # d_input = output_grad @ weights.T
        # output_grad (2,2) @ weights.T (2,3) -> (2,3)
        # For d_input[0,0]: (0.5*0.1 + 1.0*0.4) = 0.05 + 0.4 = 0.45
        # ... and so on
        expected_d_input = np.dot(output_grad, initial_weights.T)

        # Expected updated parameters
        expected_weights_updated = initial_weights - learning_rate * expected_dW
        expected_bias_updated = initial_bias - learning_rate * expected_db

        # Perform backward pass
        d_input = fc.backward(output_grad, learning_rate)

        np.testing.assert_array_almost_equal(d_input, expected_d_input, decimal=5)
        np.testing.assert_array_almost_equal(fc.weights, expected_weights_updated, decimal=5)
        np.testing.assert_array_almost_equal(fc.bias, expected_bias_updated, decimal=5)

    def test_fc_backward_gradient_check_numerical_batched(self): # Renamed and updated
        N = 2
        input_size = 2
        output_size = 1 # Keep output_size = 1 for simpler sum in loss calculation
        fc = FullyConnected(input_size, output_size)
        
        # Weights (input_size, output_size) -> (2,1)
        initial_weights = np.array([[0.05], [-0.05]]) 
        # Bias (1, output_size) -> (1,1)
        initial_bias = np.array([[0.01]])      
        fc.weights = initial_weights.copy()
        fc.bias = initial_bias.copy()

        # Input (N, input_size) -> (2,2)
        input_data = np.array([[0.8, 0.2],
                               [0.1, 0.9]])
        epsilon = 1e-5
        # d_out_mock should match the shape of the forward pass output (N, output_size)
        d_out_mock = np.ones((N, output_size)) 

        # --- Numerical dW ---
        numerical_dW = np.zeros_like(fc.weights)
        for i in range(fc.weights.shape[0]):
            for j in range(fc.weights.shape[1]):
                fc.weights = initial_weights.copy() # Ensure fresh state for each grad component
                fc.bias = initial_bias.copy()
                original_weight_val = fc.weights[i, j]
                
                fc.weights[i, j] = original_weight_val + epsilon
                loss_plus = np.sum(fc.forward(input_data) * d_out_mock) # Element-wise product then sum for loss
                
                fc.weights[i, j] = original_weight_val - epsilon
                loss_minus = np.sum(fc.forward(input_data) * d_out_mock)
                
                numerical_dW[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

        # --- Numerical db ---
        numerical_db = np.zeros_like(fc.bias)
        for i in range(fc.bias.shape[0]):
            for j in range(fc.bias.shape[1]):
                fc.weights = initial_weights.copy()
                fc.bias = initial_bias.copy()
                original_bias_val = fc.bias[i, j]
                
                fc.bias[i, j] = original_bias_val + epsilon
                loss_plus = np.sum(fc.forward(input_data) * d_out_mock)
                
                fc.bias[i, j] = original_bias_val - epsilon
                loss_minus = np.sum(fc.forward(input_data) * d_out_mock)
                
                numerical_db[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

        # --- Numerical dInput ---
        numerical_dInput = np.zeros_like(input_data)
        for n_idx in range(input_data.shape[0]): # Iterate over batch
            for k_idx in range(input_data.shape[1]): # Iterate over features
                fc.weights = initial_weights.copy()
                fc.bias = initial_bias.copy()
                original_input_val = input_data[n_idx, k_idx]
                
                input_data_plus = input_data.copy()
                input_data_plus[n_idx, k_idx] = original_input_val + epsilon
                loss_plus = np.sum(fc.forward(input_data_plus) * d_out_mock)
                
                input_data_minus = input_data.copy()
                input_data_minus[n_idx, k_idx] = original_input_val - epsilon
                loss_minus = np.sum(fc.forward(input_data_minus) * d_out_mock)
                
                numerical_dInput[n_idx, k_idx] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # --- Analytical gradients ---
        fc.weights = initial_weights.copy()
        fc.bias = initial_bias.copy()
        _ = fc.forward(input_data) # This stores fc.input_tensor
        
        # Analytical dW, db, dInput (calculated based on the logic in backward() but without updates)
        # Note: d_out_mock is used as output_grad for calculating these analytical gradients
        analytical_dW = np.dot(fc.input_tensor.T, d_out_mock) / N # Division by N for averaging
        analytical_db = np.sum(d_out_mock, axis=0, keepdims=True) / N # Division by N for averaging
        analytical_dInput = np.dot(d_out_mock, fc.weights.T)
        
        np.testing.assert_array_almost_equal(numerical_dW, analytical_dW, decimal=3, err_msg="Weight gradients don't match")
        np.testing.assert_array_almost_equal(numerical_db, analytical_db, decimal=3, err_msg="Bias gradients don't match")
        np.testing.assert_array_almost_equal(numerical_dInput, analytical_dInput, decimal=3, err_msg="Input gradients don't match")

# Remove old test names if they were just renamed, or delete if logic is now covered by batched tests
# For example, if test_fc_forward is now test_fc_forward_batched and covers the same logic but batched.

if __name__ == '__main__':
    unittest.main()