import numpy as np

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        N, C, H, W = input_tensor.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        output = np.zeros((N, C, out_h, out_w))
        self.max_indices = np.zeros((N, C, out_h, out_w, 2), dtype=int)

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        region = input_tensor[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(region)
                        output[n, c, i, j] = max_val

                        # Save indices for backward
                        max_pos_local = np.unravel_index(np.argmax(region), region.shape)
                        self.max_indices[n, c, i, j, 0] = h_start + max_pos_local[0]
                        self.max_indices[n, c, i, j, 1] = w_start + max_pos_local[1]

        return output

    def backward(self, d_out, learning_rate):
        N, C, out_h, out_w = d_out.shape # d_out shape is (N, C, out_h, out_w)
        d_input = np.zeros_like(self.input_tensor)

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_idx = self.max_indices[n, c, i, j, 0]
                        w_idx = self.max_indices[n, c, i, j, 1]
                        d_input[n, c, h_idx, w_idx] += d_out[n, c, i, j]

        return d_input
