import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, padding=0):
    C, H, W = input_data.shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    img = np.pad(input_data, ((0, 0), (padding, padding), (padding, padding)), 'constant')
    cols = np.zeros((C * filter_h * filter_w, out_h * out_w))

    col_idx = 0
    for y in range(0, H + 2 * padding - filter_h + 1, stride):
        for x in range(0, W + 2 * padding - filter_w + 1, stride):
            patch = img[:, y:y + filter_h, x:x + filter_w]
            cols[:, col_idx] = patch.flatten()
            col_idx += 1
    return cols, out_h, out_w

def col2im(cols, input_shape, filter_h, filter_w, stride=1, padding=0):
    C, H, W = input_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    img = np.zeros((C, H_padded, W_padded))
    
    out_h = (H_padded - filter_h) // stride + 1
    out_w = (W_padded - filter_w) // stride + 1

    col_idx = 0
    for y in range(0, H_padded - filter_h + 1, stride):
        for x in range(0, W_padded - filter_w + 1, stride):
            patch = cols[:, col_idx].reshape(C, filter_h, filter_w)
            img[:, y:y + filter_h, x:x + filter_w] += patch
            col_idx += 1

    return img[:, padding:H_padded - padding, padding:W_padded - padding] if padding else img
