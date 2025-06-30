import numpy as np
import unittest
from assignment_2.loss.mse import mse, mse_prime

class TestMSE(unittest.TestCase):
    def test_mse_perfect_match(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        self.assertEqual(mse(y_true, y_pred), 0.0)

    def test_mse_simple_case(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 2])
        # ( (1-2)^2 + (2-3)^2 + (3-2)^2 ) / 3
        # ( (-1)^2 + (-1)^2 + (1)^2 ) / 3
        # ( 1 + 1 + 1 ) / 3 = 3 / 3 = 1
        self.assertEqual(mse(y_true, y_pred), 1.0)

    def test_mse_different_shapes_error(self):
        # MSE typically expects y_true and y_pred to be broadcastable or same shape.
        # The current implementation relies on numpy's broadcasting.
        # If they are not broadcastable in a way that makes sense for element-wise subtraction,
        # numpy will raise an error. Let's test a case that should work with broadcasting.
        y_true = np.array([[1,1],[1,1]])
        y_pred = np.array([0,0]) # broadcast to [[0,0],[0,0]]
        # ((1-0)^2 + (1-0)^2 + (1-0)^2 + (1-0)^2) / 4 = (1+1+1+1)/4 = 1
        self.assertEqual(mse(y_true, y_pred), 1.0)

        y_true_single = np.array([1,1,1,1])
        y_pred_single = np.array([0]) # broadcast to [0,0,0,0]
        self.assertEqual(mse(y_true_single, y_pred_single), 1.0)

    def test_mse_prime_perfect_match(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        expected_grad = np.array([0, 0, 0, 0])
        np.testing.assert_array_almost_equal(mse_prime(y_true, y_pred), expected_grad, decimal=5)

    def test_mse_prime_simple_case(self):
        y_true = np.array([1, 2, 3]) # size = 3
        y_pred = np.array([2, 3, 2])
        # 2 * (y_pred - y_true) / y_true.size
        # y_pred - y_true = [1, 1, -1]
        # 2 * [1, 1, -1] / 3 = [2/3, 2/3, -2/3]
        expected_grad = np.array([2/3, 2/3, -2/3])
        np.testing.assert_array_almost_equal(mse_prime(y_true, y_pred), expected_grad, decimal=5)

    def test_mse_prime_broadcasting(self):
        y_true = np.array([[1,1],[1,1]]) # size = 4
        y_pred = np.array([0,0]) # broadcast to [[0,0],[0,0]]
        # y_pred - y_true = [[-1,-1],[-1,-1]]
        # 2 * [[-1,-1],[-1,-1]] / 4 = [[-2/4, -2/4],[-2/4, -2/4]] = [[-0.5, -0.5],[-0.5, -0.5]]
        expected_grad = np.array([[-0.5, -0.5],[-0.5, -0.5]])
        np.testing.assert_array_almost_equal(mse_prime(y_true, y_pred), expected_grad, decimal=5)

        y_true_single = np.array([1,1,1,1]) # size = 4
        y_pred_single = np.array([0]) # broadcast to [0,0,0,0]
        # y_pred - y_true = [-1,-1,-1,-1]
        # 2 * [-1,-1,-1,-1] / 4 = [-0.5, -0.5, -0.5, -0.5]
        expected_grad_single = np.array([-0.5, -0.5, -0.5, -0.5])
        np.testing.assert_array_almost_equal(mse_prime(y_true_single, y_pred_single), expected_grad_single, decimal=5)

if __name__ == '__main__':
    unittest.main()