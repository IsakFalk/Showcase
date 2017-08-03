# Get the mnist data

import part2lib as p2l
from part2lib import Data
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

# print input_im.shape

# plt.subplot(131)
# plt.imshow(input_im[0, 0, :, :])
# plt.subplot(132)
# plt.imshow(input_im[1, 0, :, :])

# plt.show()

# im2col and other functions

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

# Convolutional layer

def conv_forward(X, W, b, stride=1, padding=1):
    cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = np.dot(W_col, X_col) + b
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, W, b, stride, padding, X_col)

    return out, cache


def conv_backward(dout, cache):
    X, W, b, stride, padding, X_col = cache
    n_filter, d_filter, h_filter, w_filter = W.shape

    db = np.sum(dout, axis=(0, 2, 3))
    db = db.reshape(n_filter, -1)

    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    dW = np.dot(dout_reshaped, X_col.T)
    dW = dW.reshape(W.shape)

    W_reshape = W.reshape(n_filter, -1)
    dX_col = np.dot(W_reshape.T, dout_reshaped)
    dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

    return dX, dW, db


# max pool layer

def _pool_forward(X, pool_fun, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1

    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

    out, pool_cache = pool_fun(X_col)

    out = out.reshape(h_out, w_out, n, d)
    out = out.transpose(2, 3, 0, 1)

    cache = (X, size, stride, X_col, pool_cache)

    return out, cache


def _pool_backward(dout, dpool_fun, cache):
    X, size, stride, X_col, pool_cache = cache
    n, d, w, h = X.shape

    dX_col = np.zeros_like(X_col)
    dout_col = dout.transpose(2, 3, 0, 1).ravel()

    dX = dpool_fun(dX_col, dout_col, pool_cache)

    dX = col2im_indices(dX_col, (n * d, 1, h, w), size, size, padding=0, stride=stride)
    dX = dX.reshape(X.shape)

    return dX


def maxpool_forward(X, size=2, stride=2):
    def maxpool(X_col):
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        return out, max_idx

    return _pool_forward(X, maxpool, size, stride)


def maxpool_backward(dout, cache):
    def dmaxpool(dX_col, dout_col, pool_cache):
        dX_col[pool_cache, range(dout_col.size)] = dout_col
        return dX_col

    return _pool_backward(dout, dmaxpool, cache)


class Flattener(Layer):

    def __init__(self, heigth, width, channels):
        """
        Flattens an image batch-tensor to a design matrix. The image batch-tensor follows the dimensionality convention:
        batch x height x width x channels

        Args:
            heigth:
            width:
            channels:
        """
        self.b = 0
        self.h = heigth
        self.w = width
        self.c = channels

    def init_layer(self, n_inputs):
        return self.c * self.h * self.w

    def forward_pass(self, layer_input):
        # save shape to convert the backward message into image batch-tensor convention
        self.b, self.c, self.h, self.w = layer_input.shape
        return layer_input.reshape(self.b, self.h * self.w * self.c)

    def backward_pass(self, dl_dy, optimiser):
        return dl_dy.reshape(self.b, self.c, self.h, self.w)


if __name__ == '__main__':
    """Test the different passes"""

    # Needed data

    data = Data(1)

    # need to sort out input data

    X, _ = data.next_batch()
    X, _ = data.next_batch()
    X, _ = data.next_batch()

    input_size = 784
    output_size = 10
    batch_size = 200
    img_width = 28
    img_height = 28
    channels = 1
    filters = 16

    # shape of the batched images, resized
    image_shape = (batch_size, channels, img_height, img_width)
    filter_shape = (batch_size, filters, img_height/2, img_width/2)
    flatten_output = (batch_size, filters * img_height/4 * img_width/4)

    layers = [p2l.reshape_input(),
              p2l.conv2d(image_shape),
              p2l.max_pooling(),
              p2l.conv2d(filter_shape),
              p2l.max_pooling(),
              p2l.flatten(),
              p2l.ReLU(),
              p2l.OutputLayer(flatten_output[1], 10)]

    nn = p2l.NeuralNetwork(layers, 0.01, 1, 200, 'd_test')

    nn.feed_forward


    # Imported from part2lib

    # # conv2d layer

    # conv2d = p2l.conv2d(X.shape)

    # # forward pass

    # FX = conv2d.forward_pass(X)

    # # backward pass

    # conv2d.backward_pass(0.1, FX, 0)

    # # max_pool layer

    # max_pooling = p2l.max_pooling()

    # (mpX, _) = max_pooling.forward_pass(X)
    # (mpX, _) = max_pooling.forward_pass(mpX)

    # # plot it

    # plt.subplot(121)
    # plt.imshow(X[0,0,:,:], interpolation='none')
    # plt.subplot(122)
    # plt.imshow(mpX[0,0,:,:], interpolation='none')
    # plt.show()

