# model.py
import numpy as np
import os

# 复制必要的函数
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def im2col(input_data, filter_h, filter_w):
    N, H, W, C = input_data.shape
    out_h = H - filter_h + 1
    out_w = W - filter_w + 1
    col = np.zeros((N, out_h * out_w, filter_h * filter_w * C))
    for y in range(out_h):
        for x in range(out_w):
            patch = input_data[:, y:y+filter_h, x:x+filter_w, :]
            col[:, y*out_w + x, :] = patch.reshape(N, -1)
    return col

# 你的模型结构（删掉训练用的 dropout、backward等）
class SimpleCNN:
    def __init__(self, input_shape=(32,32,3), num_classes=2):
        self.filters1 = np.zeros((3, 3, 3, 32))
        self.filters2 = np.zeros((3, 3, 32, 64))
        self.fc_weights1 = np.zeros((2304, 128))
        self.fc_weights2 = np.zeros((128, num_classes))

    def conv2d(self, x, filters):
        b, h, w, c = x.shape
        f_h, f_w, in_c, out_c = filters.shape
        col = im2col(x, f_h, f_w)
        filters_col = filters.reshape(-1, out_c)
        out = np.dot(col, filters_col)
        out_h = h - f_h + 1
        out_w = w - f_w + 1
        return out.reshape(b, out_h, out_w, out_c)

    def max_pool(self, x, size=2):
        b, h, w, c = x.shape
        out_h, out_w = h // size, w // size
        out = np.zeros((b, out_h, out_w, c))
        for i in range(out_h):
            for j in range(out_w):
                region = x[:, i*size:i*size+size, j*size:j*size+size, :]
                out[:, i, j, :] = np.max(region, axis=(1,2))
        return out

    def flatten(self, x):
        return x.reshape(x.shape[0], -1)

    def forward(self, X):
        c1 = self.conv2d(X, self.filters1)
        r1 = relu(c1)
        p1 = self.max_pool(r1)

        c2 = self.conv2d(p1, self.filters2)
        r2 = relu(c2)
        p2 = self.max_pool(r2)

        flat = self.flatten(p2)
        fc1 = relu(np.dot(flat, self.fc_weights1))
        fc2 = np.dot(fc1, self.fc_weights2)
        out = softmax(fc2)
        return out

    def load(self, folder='model'):
        self.filters1 = np.load(os.path.join(folder, 'filters1.npy'))
        self.filters2 = np.load(os.path.join(folder, 'filters2.npy'))
        self.fc_weights1 = np.load(os.path.join(folder, 'fc_weights1.npy'))
        self.fc_weights2 = np.load(os.path.join(folder, 'fc_weights2.npy'))
