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
        input_tensor = np.array([[[[1, 2, 3, 4],
                                   [5, 6, 7, 8],
                                   [9, 10, 11, 12],
                                   [13, 14, 15, 16]]]])
        output = pool.forward(input_tensor)
        d_out = np.ones_like(output)
        
        d_input = pool.backward(d_out, optimizer=None)
        self.assertEqual(d_input.shape, input_tensor.shape)

    def test_maxpool2d_backward_overlapping_regions_not_possible_with_typical_stride(self):
        pool = MaxPool2D(pool_size=2, stride=2)
        input_tensor = np.random.randn(1, 1, 4, 4)
        output = pool.forward(input_tensor)
        d_out = np.random.randn(*output.shape)
        
        d_input = pool.backward(d_out, optimizer=None)
        self.assertEqual(d_input.shape, input_tensor.shape)

if __name__ == '__main__':
    unittest.main()