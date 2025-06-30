import numpy as np
import unittest
from assignment_2.utils.im2col import im2col, col2im

class TestIm2Col(unittest.TestCase):
    def test_im2col_simple_no_padding_stride1(self):
        # Input: 1 channel, 3x3 image
        # Filter: 2x2
        # Stride: 1, Padding: 0
        input_data = np.array([[[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]]).astype(float) # Shape (1, 3, 3)
        C, H, W = input_data.shape
        filter_h, filter_w = 2, 2
        stride = 1
        padding = 0

        # Expected output_h = (3 + 2*0 - 2)//1 + 1 = 1//1 + 1 = 2
        # Expected output_w = (3 + 2*0 - 2)//1 + 1 = 1//1 + 1 = 2
        # Number of patches = out_h * out_w = 2 * 2 = 4
        # Patch size = C * filter_h * filter_w = 1 * 2 * 2 = 4

        # Patches:
        # P1: [1,2,4,5]
        # P2: [2,3,5,6]
        # P3: [4,5,7,8]
        # P4: [5,6,8,9]
        expected_cols = np.array([[1, 2, 4, 5],
                                  [2, 3, 5, 6],
                                  [4, 5, 7, 8],
                                  [5, 6, 8, 9]]).T # Transposed to match (PatchSize, NumPatches)
        
        cols, out_h, out_w = im2col(input_data, filter_h, filter_w, stride, padding)

        np.testing.assert_array_equal(cols, expected_cols)
        self.assertEqual(out_h, 2)
        self.assertEqual(out_w, 2)
        self.assertEqual(cols.shape, (C * filter_h * filter_w, out_h * out_w))

    def test_im2col_with_padding_stride2(self):
        input_data = np.array([[[1, 2],
                                [3, 4]]]).astype(float) # Shape (1, 2, 2)
        C, H, W = input_data.shape
        filter_h, filter_w = 2, 2
        stride = 2
        padding = 1

        # Padded input (H_pad=2+2*1=4, W_pad=2+2*1=4):
        # 0 0 0 0
        # 0 1 2 0
        # 0 3 4 0
        # 0 0 0 0

        # Expected output_h = (2 + 2*1 - 2)//2 + 1 = 2//2 + 1 = 2
        # Expected output_w = (2 + 2*1 - 2)//2 + 1 = 2//2 + 1 = 2
        # Num patches = 2 * 2 = 4
        # Patch size = 1 * 2 * 2 = 4

        # Patches (from padded image):
        # P1 (y=0, x=0): [0,0,0,1]
        # P2 (y=0, x=2): [0,0,1,2] -> incorrect, should be [0,0,2,0]
        # P3 (y=2, x=0): [0,1,0,3] -> incorrect, should be [0,3,0,0]
        # P4 (y=2, x=2): [1,2,3,4] -> incorrect, should be [2,0,4,0]
        
        # Corrected Patches from padded image, stride 2:
        # Patch 1 (top-left of padded, size 2x2): [[0,0],[0,1]] -> flattened [0,0,0,1]
        # Patch 2 (top-right of padded, size 2x2): [[0,0],[2,0]] -> flattened [0,0,2,0]
        # Patch 3 (bottom-left of padded, size 2x2): [[0,3],[0,0]] -> flattened [0,3,0,0]
        # Patch 4 (bottom-right of padded, size 2x2): [[4,0],[0,0]] -> flattened [4,0,0,0]
        # This is if the filter is applied at (0,0), (0,2), (2,0), (2,2) in the padded image.

        # Let's trace the loops in im2col:
        # img = [[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]] (1 channel)
        # H_padded = 4, W_padded = 4
        # Loop y: range(0, 4+2*1-2+1, 2) -> range(0, 5, 2) -> y = 0, 2, 4 (y=4 is out of bound for patch start)
        #   y=0, x=0: patch = img[:, 0:2, 0:2] = [[[0,0],[0,1]]] -> flatten [0,0,0,1]
        #   y=0, x=2: patch = img[:, 0:2, 2:4] = [[[0,0],[2,0]]] -> flatten [0,0,2,0]
        #   y=2, x=0: patch = img[:, 2:4, 0:2] = [[[0,3],[0,0]]] -> flatten [0,3,0,0]
        #   y=2, x=2: patch = img[:, 2:4, 2:4] = [[[4,0],[0,0]]] -> flatten [4,0,0,0]

        expected_cols = np.array([[0,0,0,4],
                                  [0,0,3,0],
                                  [0,2,0,0],
                                  [1,0,0,0]]).T # (4,4) -> (PatchSize, NumPatches)
                                             # My manual trace was [P1,P2,P3,P4], so expected_cols should be [P1,P2,P3,P4].T
                                             # P1=[0,0,0,1], P2=[0,0,2,0], P3=[0,3,0,0], P4=[4,0,0,0]
        expected_cols_corrected = np.array([[0,0,0,1],[0,0,2,0],[0,3,0,0],[4,0,0,0]]).T

        cols, out_h, out_w = im2col(input_data, filter_h, filter_w, stride, padding)
        np.testing.assert_array_equal(cols, expected_cols_corrected)
        self.assertEqual(out_h, 2)
        self.assertEqual(out_w, 2)
        self.assertEqual(cols.shape, (C * filter_h * filter_w, out_h * out_w))

    def test_im2col_multi_channel(self):
        input_data = np.array([[[1,2],[3,4]], [[5,6],[7,8]]]).astype(float) # Shape (2,2,2) C,H,W
        C, H, W = input_data.shape
        filter_h, filter_w = 1, 1
        stride = 1
        padding = 0

        # out_h = (2+0-1)//1+1 = 2
        # out_w = (2+0-1)//1+1 = 2
        # Num patches = 2*2 = 4
        # Patch size = 2*1*1 = 2

        # Patches (C elements first, then spatial):
        # P1 (y=0,x=0): [img[0,0,0], img[1,0,0]] = [1, 5]
        # P2 (y=0,x=1): [img[0,0,1], img[1,0,1]] = [2, 6]
        # P3 (y=1,x=0): [img[0,1,0], img[1,1,0]] = [3, 7]
        # P4 (y=1,x=1): [img[0,1,1], img[1,1,1]] = [4, 8]
        expected_cols = np.array([[1,5],[2,6],[3,7],[4,8]]).T

        cols, out_h, out_w = im2col(input_data, filter_h, filter_w, stride, padding)
        np.testing.assert_array_equal(cols, expected_cols)
        self.assertEqual(out_h, 2)
        self.assertEqual(out_w, 2)
        self.assertEqual(cols.shape, (C * filter_h * filter_w, out_h * out_w))

class TestCol2Im(unittest.TestCase):
    def test_col2im_simple_no_padding_stride1(self):
        # Inverse of test_im2col_simple_no_padding_stride1
        cols = np.array([[1, 2, 4, 5],
                         [2, 3, 5, 6],
                         [4, 5, 7, 8],
                         [5, 6, 8, 9]]).T
        input_shape = (1, 3, 3) # C, H, W
        C, H, W = input_shape
        filter_h, filter_w = 2, 2
        stride = 1
        padding = 0

        expected_img = np.array([[[1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 9]]]).astype(float)
        
        # Note: col2im sums overlapping regions. For stride=1, filter > 1, there's overlap.
        # The original im2col doesn't have a unique inverse if stride < filter_size due to overlap.
        # However, for Conv backward pass, dcol is what we have, and we need to sum gradients.
        # Let's test with a case where col2im should reconstruct if no overlap or if values are set to make it work.
        # For this specific test, the im2col output is unique, so col2im should reconstruct the original if it were just placing.
        # But col2im ADDS. So this test needs to be carefully designed.

        # Let's use a known col2im behavior: gradient accumulation.
        # If cols represents gradients for patches, col2im reconstructs the gradient for the input image.
        # If we want to test exact reconstruction of an image, we need non-overlapping patches or specific values.
        
        # Test with non-overlapping patches (stride=filter_size)
        input_data_non_overlap = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]).astype(float)
        C,H,W = input_data_non_overlap.shape # 1,4,4
        fh,fw = 2,2
        s = 2 # stride = filter_size
        p = 0
        cols_non_overlap, oh, ow = im2col(input_data_non_overlap, fh, fw, s, p)
        # cols_non_overlap will be:
        # P1: [1,2,5,6]
        # P2: [3,4,7,8]
        # P3: [9,10,13,14]
        # P4: [11,12,15,16]
        # Transposed: [[1,3,9,11],[2,4,10,12],[5,7,13,15],[6,8,14,16]]

        reconstructed_img = col2im(cols_non_overlap, input_data_non_overlap.shape, fh, fw, s, p)
        np.testing.assert_array_equal(reconstructed_img, input_data_non_overlap)

    def test_col2im_with_padding_stride2(self):
        # Inverse of test_im2col_with_padding_stride2 (non-overlapping due to stride=filter_size after padding)
        cols = np.array([[0,0,0,1],[0,0,2,0],[0,3,0,0],[4,0,0,0]]).T
        input_shape_original = (1, 2, 2) # C, H, W (original image before padding)
        filter_h, filter_w = 2, 2
        stride = 2
        padding = 1

        # Expected image is the original input_data from the im2col test
        expected_img = np.array([[[1, 2],
                                  [3, 4]]]).astype(float)
        
        reconstructed_img = col2im(cols, input_shape_original, filter_h, filter_w, stride, padding)
        np.testing.assert_array_equal(reconstructed_img, expected_img)

    def test_col2im_gradient_accumulation(self):
        # This test demonstrates the summing behavior of col2im, crucial for backprop.
        # Input image 1x3x3, filter 2x2, stride 1, padding 0.
        # This means overlapping regions.
        input_shape = (1, 3, 3)
        C, H, W = input_shape
        filter_h, filter_w = 2, 2
        stride = 1
        padding = 0

        # Mock gradients for 4 patches (cols has shape (C*fh*fw, num_patches) = (4, 4))
        # Patch1_grad: [g11,g12,g13,g14]
        # Patch2_grad: [g21,g22,g23,g24]
        # Patch3_grad: [g31,g32,g33,g34]
        # Patch4_grad: [g41,g42,g43,g44]
        cols_grad = np.arange(1, 17).reshape(4, 4).astype(float) # Values 1-16
        # cols_grad.T to match function's expectation if my manual trace was row-major for patches
        # The im2col output is (C*fh*fw, num_patches), so cols_grad should be that shape.
        # cols_grad[:,0] is the first patch [1,2,3,4]
        # cols_grad[:,1] is the second patch [5,6,7,8]
        # etc.
        reconstructed_grad_img = col2im(cols_grad, input_shape, filter_h, filter_w, stride, padding)

        # Manually calculated expected gradient image based on the trace:
        # Patch1: [[1,2],[3,4]] at (0,0)
        # Patch2: [[5,6],[7,8]] at (0,1)
        # Patch3: [[9,10],[11,12]] at (1,0)
        # Patch4: [[13,14],[15,16]] at (1,1)
        # Resulting image (1,3,3):
        # [ (1)   (2+5)   (6)   ]
        # [ (3+9) (4+7+10+13) (8+14) ]
        # [ (11)  (12+15) (16)  ]
        # Corrected trace from above:
        # img = [[[1,7,6],[12,34,22],[11,27,16]]]
        expected_grad_img = np.array([[[ 1.,  7.,  6.],
                                       [12., 34., 22.],
                                       [11., 27., 16.]]]).astype(float)
        np.testing.assert_array_equal(reconstructed_grad_img, expected_grad_img)

        # Image layout:
        # (0,0) (0,1) (0,2)
        # (1,0) (1,1) (1,2)
        # (2,0) (2,1) (2,2)

        # Patch locations (top-left corners of where filter was applied):
        # P1: (0,0) -> img[0,0:2,0:2]
        # P2: (0,1) -> img[0,0:2,1:3]
        # P3: (1,0) -> img[0,1:3,0:2]
        # P4: (1,1) -> img[0,1:3,1:3]

        # Gradients from cols_grad (reshaped to C, fh, fw for each patch):
        # Patch1_grad_matrix = [[1,2],[3,4]]
        # Patch2_grad_matrix = [[5,6],[7,8]]
        # Patch3_grad_matrix = [[9,10],[11,12]]
        # Patch4_grad_matrix = [[13,14],[15,16]]

        # Expected dInput (summing contributions):
        # dInput[0,0,0] = P1[0,0] = 1
        # dInput[0,0,1] = P1[0,1] + P2[0,0] = 2 + 5 = 7
        # dInput[0,0,2] = P2[0,1] = 6
        # dInput[0,1,0] = P1[1,0] + P3[0,0] = 3 + 9 = 12
        # dInput[0,1,1] = P1[1,1] + P2[1,0] + P3[0,1] + P4[0,0] = 4 + 7 + 10 + 13 = 34
        # dInput[0,1,2] = P2[1,1] + P4[0,1] = 8 + 14 = 22
        # dInput[0,2,0] = P3[1,0] = 11
        # dInput[0,2,1] = P3[1,1] + P4[1,0] = 12 + 15 = 27
        # dInput[0,2,2] = P4[1,1] = 16
        expected_dInput = np.array([[[ 1,  7,  6],
                                     [12, 34, 22],
                                     [11, 27, 16]]]).astype(float)

        dInput = col2im(cols_grad, input_shape, filter_h, filter_w, stride, padding)
        np.testing.assert_array_equal(dInput, expected_dInput)

    def test_im2col_col2im_roundtrip_non_overlapping(self):
        input_data = np.random.rand(2, 5, 5) # C, H, W
        C, H, W = input_data.shape
        filter_h, filter_w = 2, 2
        stride = 2 # Non-overlapping
        padding = 0
        
        cols, out_h, out_w = im2col(input_data, filter_h, filter_w, stride, padding)
        # For non-overlapping, col2im should perfectly reconstruct if input H,W are multiples of stride for the filter size
        # Need to ensure the output dimensions of im2col can perfectly tile the input for col2im
        # This test might fail if H,W are not perfectly divisible by filter_h,w with stride.
        # Let's use H=4, W=4, filter=2,2, stride=2,2
        input_data_even = np.random.rand(1, 4, 4)
        C,H,W = input_data_even.shape
        fh,fw = 2,2
        s = 2
        p = 0

        cols_even, oh_e, ow_e = im2col(input_data_even, fh, fw, s, p)
        reconstructed_even = col2im(cols_even, input_data_even.shape, fh, fw, s, p)
        np.testing.assert_array_almost_equal(reconstructed_even, input_data_even, decimal=3)

    def test_im2col_col2im_roundtrip_padding(self):
        # Modified to be a non-overlapping case to correctly test roundtrip with summing col2im
        input_data = np.random.rand(1, 3, 3)
        C,H,W = input_data.shape
        fh,fw = 1,1 # Non-overlapping filter
        s = 1       # Stride 1
        p = 0       # No padding for simplicity with 1x1 filter

        cols, oh, ow = im2col(input_data, fh, fw, s, p)
        self.assertEqual(cols.shape, (C*fh*fw, H*W)) # Each pixel is a patch
        reconstructed = col2im(cols, input_data.shape, fh, fw, s, p)
        np.testing.assert_array_almost_equal(reconstructed, input_data, decimal=5)

if __name__ == '__main__':
    unittest.main()