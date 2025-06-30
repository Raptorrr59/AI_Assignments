import numpy as np
import unittest
from assignment_2.layers.maxpooling import MaxPool2D

class TestMaxPool2D(unittest.TestCase):
    def test_maxpool2d_initialization(self):
        pool = MaxPool2D(pool_size=3, stride=1)
        self.assertEqual(pool.pool_size, 3)
        self.assertEqual(pool.stride, 1)

        pool_default = MaxPool2D()
        self.assertEqual(pool_default.pool_size, 2)
        self.assertEqual(pool_default.stride, 2)

    def test_maxpool2d_forward_simple(self):
        pool = MaxPool2D(pool_size=2, stride=2)
        input_data = np.array([[[[1, 2, 3, 4],
                                 [5, 6, 7, 8],
                                 [9, 10, 11, 12],
                                 [13, 14, 15, 16]]]]).astype(float) # (1, 1, 4, 4)
        
        # Expected output:
        # Max of [1,2,5,6] = 6
        # Max of [3,4,7,8] = 8
        # Max of [9,10,13,14] = 14
        # Max of [11,12,15,16] = 16
        expected_output = np.array([[[[6, 8],
                                      [14, 16]]]]).astype(float) # Shape (1,1,2,2)
        output = pool.forward(input_data)
        np.testing.assert_array_equal(output, expected_output)
        self.assertEqual(output.shape, (1, 1, 2, 2)) # (N, C, out_h, out_w)

        # Argmax check removed as self.argmax is replaced by self.max_indices

    def test_maxpool2d_forward_stride_one(self):
        pool = MaxPool2D(pool_size=2, stride=1)
        input_data = np.array([[[[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9]]]]).astype(float) # (1,1,3,3)
        # Output H = (3 - 2)//1 + 1 = 2
        # Output W = (3 - 2)//1 + 1 = 2
        # Expected output:
        # Max of [1,2,4,5] = 5
        # Max of [2,3,5,6] = 6
        # Max of [4,5,7,8] = 8
        # Max of [5,6,8,9] = 9
        expected_output = np.array([[[[5, 6],
                                      [8, 9]]]]).astype(float) # Shape (1,1,2,2)
        output = pool.forward(input_data)
        np.testing.assert_array_equal(output, expected_output)
        self.assertEqual(output.shape, (1, 1, 2, 2))

        # Argmax check removed as self.argmax is replaced by self.max_indices

    def test_maxpool2d_forward_multi_channel(self):
        pool = MaxPool2D(pool_size=2, stride=2)
        input_data = np.array([[[[1, 2], [3, 4]], # Channel 1
                                [[5, 6], [7, 8]]]]).astype(float) # Channel 2. Shape (1,2,2,2)
                                                                    # Actually (N, C, H, W), so (1,2,2,2)
        # For N=1, C=2, H=2, W=2. pool_size=2, stride=2. Output H,W = (2-2)//2+1 = 1
        # Channel 1: Max of [1,2,3,4] = 4
        # Channel 2: Max of [5,6,7,8] = 8
        expected_output = np.array([[[[4.]], [[8.]]]]).astype(float) # Shape (1,2,1,1)
        output = pool.forward(input_data)
        np.testing.assert_array_equal(output, expected_output)
        self.assertEqual(output.shape, (1, 2, 1, 1)) # (N, C, out_h, out_w)

        # Argmax check removed as self.argmax is replaced by self.max_indices

    def test_maxpool2d_backward_simple(self):
        pool = MaxPool2D(pool_size=2, stride=2)
        input_data = np.array([[[[1, 2, 3, 4],
                                 [5, 6, 7, 8],
                                 [9, 10, 11, 12],
                                 [13, 14, 15, 16]]]]).astype(float) # Shape (1,1,4,4)
        _ = pool.forward(input_data) # To set self.input_tensor and self.max_indices

        d_out = np.array([[[[0.1, 0.2],
                            [0.3, 0.4]]]]).astype(float) # Shape (1,1,2,2)
        
        expected_d_input = np.zeros_like(input_data) # Shape (N,C,H,W) -> (1,1,4,4)
        # d_out[0,0,0,0] = 0.1 corresponds to output 6 (input_data[0,0,1,1])
        # d_out[0,0,0,1] = 0.2 corresponds to output 8 (input_data[0,0,1,3])
        # d_out[0,0,1,0] = 0.3 corresponds to output 14 (input_data[0,0,3,1])
        # d_out[0,0,1,1] = 0.4 corresponds to output 16 (input_data[0,0,3,3])

        expected_d_input[0, 0, 1, 1] = 0.1
        expected_d_input[0, 0, 1, 3] = 0.2
        expected_d_input[0, 0, 3, 1] = 0.3
        expected_d_input[0, 0, 3, 3] = 0.4

        d_input = pool.backward(d_out, learning_rate=0.01)
        np.testing.assert_array_almost_equal(d_input, expected_d_input, decimal=5)
        self.assertEqual(d_input.shape, input_data.shape)

    def test_maxpool2d_backward_overlapping_regions_not_possible_with_typical_stride(self):
        # MaxPool with stride equal to pool_size doesn't have overlapping regions for gradient distribution.
        # If stride < pool_size, then it's possible, but the current backward pass sums gradients, which is correct.
        # This test is more of a conceptual check.
        pool = MaxPool2D(pool_size=2, stride=1)
        # Let's make one 8 slightly larger to test unique argmax propagation
        input_data_mod = np.array([[[[1, 5, 2],
                                     [6, 3, 7],
                                     [2, 8.1, 4]]]]).astype(float) # Shape (1,1,3,3)
        _ = pool.forward(input_data_mod)
        # Output for N=0, C=0: [[6,7],[8.1,8.1]]
        # Argmax for 6: input_data_mod[0,0,1,0]
        # Argmax for 7: input_data_mod[0,0,1,2]
        # Argmax for first 8.1: input_data_mod[0,0,2,1]
        # Argmax for second 8.1: input_data_mod[0,0,2,1]
        # The argmax will be set for input_data_mod[0,0,2,1] for both output cells [0,0,1,0] and [0,0,1,1]
        # This means the gradient from d_out[0,0,1,0] and d_out[0,0,1,1] will both go to input_data_mod[0,0,2,1]

        d_out = np.array([[[[0.1, 0.2],
                            [0.3, 0.4]]]]).astype(float) # Shape (1,1,2,2)
        
        expected_d_input = np.zeros_like(input_data_mod) # Shape (N,C,H,W) -> (1,1,3,3)
        expected_d_input[0,0,1,0] = 0.1 # from d_out[0,0,0,0] for 6 (N=0,C=0, H=1, W=0)
        expected_d_input[0,0,1,2] = 0.2 # from d_out[0,0,0,1] for 7 (N=0,C=0, H=1, W=2)
        expected_d_input[0,0,2,1] = 0.3 + 0.4 # from d_out[0,0,1,0] and d_out[0,0,1,1] for 8.1 (N=0,C=0, H=2, W=1)

        d_input = pool.backward(d_out, learning_rate=0.01)
        np.testing.assert_array_almost_equal(d_input, expected_d_input, decimal=5)

if __name__ == '__main__':
    unittest.main()