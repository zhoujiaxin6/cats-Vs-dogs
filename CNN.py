import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# 参数设置
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
TRAIN_TEST_SPLIT = 0.8
MAX_IMAGES = 1000
DROPOUT_RATE = 0.2

# 数据加载与预处理
def load_images_from_folder(folder, label, image_size=IMAGE_SIZE, max_images=MAX_IMAGES):
    images, labels = [], []
    count = 0
    for filename in os.listdir(folder):
        if count >= max_images:
            break
        try:
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB').resize(image_size)
            images.append(np.array(img) / 255.0)
            labels.append(label)
            count += 1
        except:
            continue
    return images, labels

cat_images, cat_labels = load_images_from_folder('datasets/cats', 0)
dog_images, dog_labels = load_images_from_folder('datasets/dogs', 1)

images = np.array(cat_images + dog_images)
labels = np.array(cat_labels + dog_labels)
combined = list(zip(images, labels))
random.shuffle(combined)
images, labels = zip(*combined)

split = int(len(images) * TRAIN_TEST_SPLIT)
X_train, X_test = np.array(images[:split]), np.array(images[split:])
y_train, y_test = np.array(labels[:split]), np.array(labels[split:])

# 工具函数
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def cross_entropy_loss(pred, true):
    return -np.sum(true * np.log(pred + 1e-8)) / pred.shape[0]

def dropout(x, rate):
    mask = (np.random.rand(*x.shape) > rate).astype(float)
    return x * mask, mask

# im2col 用于加速卷积
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

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads, params):
        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# 模型定义
class SimpleCNN:
    def __init__(self, input_shape, num_classes=2):
        self.filters1 = np.random.randn(3, 3, 3, 32) * 0.1
        self.filters2 = np.random.randn(3, 3, 32, 64) * 0.1
        self.fc_weights1 = np.random.randn(2304, 128) * 0.1
        self.fc_weights2 = np.random.randn(128, num_classes) * 0.1

        # 初始化 Adam 优化器
        self.optimizer = Adam([
            self.filters1,
            self.filters2,
            self.fc_weights1,
            self.fc_weights2
        ], lr=LEARNING_RATE)

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
        self.pool_mask = np.zeros_like(x)
        for i in range(out_h):
            for j in range(out_w):
                region = x[:, i*size:i*size+size, j*size:j*size+size, :]
                max_val = np.max(region, axis=(1,2))
                out[:, i, j, :] = max_val
                for bi in range(b):
                    for ci in range(c):
                        region_slice = region[bi, :, :, ci]
                        max_pos = np.unravel_index(np.argmax(region_slice), region_slice.shape)
                        self.pool_mask[bi, i*size+max_pos[0], j*size+max_pos[1], ci] = 1
        return out

    def max_pool_backward(self, d_out, orig_input, size=2):
        d_input = np.zeros_like(orig_input)
        b, h, w, c = orig_input.shape
        out_h, out_w = d_out.shape[1], d_out.shape[2]
        for i in range(out_h):
            for j in range(out_w):
                for bi in range(b):
                    for ci in range(c):
                        idx = np.argmax(orig_input[bi, i*size:i*size+size, j*size:j*size+size, ci])
                        ii, jj = np.unravel_index(idx, (size, size))
                        d_input[bi, i*size+ii, j*size+jj, ci] = d_out[bi, i, j, ci]
        return d_input

    def flatten(self, x):
        return x.reshape(x.shape[0], -1)

    def forward(self, X, training=True):
        self.X = X
        self.c1 = self.conv2d(X, self.filters1)
        self.r1 = relu(self.c1)
        self.p1 = self.max_pool(self.r1)

        self.c2 = self.conv2d(self.p1, self.filters2)
        self.r2 = relu(self.c2)
        self.p2 = self.max_pool(self.r2)

        self.flat = self.flatten(self.p2)
        self.fc1 = np.dot(self.flat, self.fc_weights1)
        self.rfc1 = relu(self.fc1)
        if training:
            self.rfc1, self.dropout_mask = dropout(self.rfc1, DROPOUT_RATE)
        self.fc2 = np.dot(self.rfc1, self.fc_weights2)
        self.out = softmax(self.fc2)
        return self.out

    def backward(self, y_true, learning_rate=LEARNING_RATE):
        batch_size = y_true.shape[0]
        error = (self.out - y_true) / batch_size

        d_fc_weights2 = np.dot(self.rfc1.T, error)
        d_rfc1 = np.dot(error, self.fc_weights2.T) * relu_derivative(self.fc1)
        d_rfc1 *= self.dropout_mask

        d_fc_weights1 = np.dot(self.flat.T, d_rfc1)

        d_flat = np.dot(d_rfc1, self.fc_weights1.T)
        d_p2 = d_flat.reshape(self.p2.shape)

        d_r2 = self.max_pool_backward(d_p2, self.r2)
        d_c2 = d_r2 * relu_derivative(self.c2)

        d_filters2 = np.zeros_like(self.filters2)
        for i in range(d_c2.shape[1]):
            for j in range(d_c2.shape[2]):
                region = self.p1[:, i:i+3, j:j+3, :]
                for k in range(64):
                    d_filters2[..., k] += np.sum(region * d_c2[:, i:i+1, j:j+1, k:k+1], axis=0)

        d_r1 = self.max_pool_backward(np.zeros_like(self.p1), self.r1)
        d_c1 = d_r1 * relu_derivative(self.c1)

        d_filters1 = np.zeros_like(self.filters1)
        for i in range(d_c1.shape[1]):
            for j in range(d_c1.shape[2]):
                region = self.X[:, i:i+3, j:j+3, :]
                for k in range(32):
                    d_filters1[..., k] += np.sum(region * d_c1[:, i:i+1, j:j+1, k:k+1], axis=0)

        grads = [
            d_filters1 / batch_size,
            d_filters2 / batch_size,
            d_fc_weights1,
            d_fc_weights2
        ]

        params = [
            self.filters1,
            self.filters2,
            self.fc_weights1,
            self.fc_weights2
        ]

        self.optimizer.step(grads, params)

    def save(self, folder='model'):
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, 'filters1.npy'), self.filters1)
        np.save(os.path.join(folder, 'filters2.npy'), self.filters2)
        np.save(os.path.join(folder, 'fc_weights1.npy'), self.fc_weights1)
        np.save(os.path.join(folder, 'fc_weights2.npy'), self.fc_weights2)

    def load(self, folder='model'):
        self.filters1 = np.load(os.path.join(folder, 'filters1.npy'))
        self.filters2 = np.load(os.path.join(folder, 'filters2.npy'))
        self.fc_weights1 = np.load(os.path.join(folder, 'fc_weights1.npy'))
        self.fc_weights2 = np.load(os.path.join(folder, 'fc_weights2.npy'))

# 训练函数
def train(model, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE):
    losses = []
    accuracies = []
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            y_one_hot = np.zeros((batch_y.size, 2))
            y_one_hot[np.arange(batch_y.size), batch_y] = 1
            out = model.forward(batch_X)
            loss = cross_entropy_loss(out, y_one_hot)
            total_loss += loss

            predictions = np.argmax(out, axis=1)
            correct_predictions += np.sum(predictions == batch_y)
            total_predictions += batch_y.size

            model.backward(y_one_hot)

        avg_loss = total_loss / (len(X_train) / batch_size)
        accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        losses.append(avg_loss)
        accuracies.append(accuracy)
    return losses, accuracies

# 评估函数
def evaluate(model, X_test, y_test, batch_size=32):
    y_pred = []
    y_score = []  # 用于 ROC AUC

    for i in range(0, len(X_test), batch_size):
        batch_X = X_test[i:i + batch_size]
        out = model.forward(batch_X, training=False)
        batch_pred = np.argmax(out, axis=1)
        y_pred.extend(batch_pred)

        if out.shape[1] == 2:  # 二分类用于 ROC AUC
            y_score.extend(out[:, 1])  # 第二列是狗的概率

    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    print("=== 分类报告 ===")
    print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

    print("=== 总体准确率 ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("=== 混淆矩阵 ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 混淆矩阵可视化
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# 主流程
model = SimpleCNN(input_shape=(32, 32, 3))
losses, accuracies = train(model, X_train, y_train)
model.save()
evaluate(model, X_test, y_test)

# 绘制损失和准确率图像
fig, ax1 = plt.subplots()

# 绘制损失值
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:red')
ax1.plot(losses, color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

# 创建第二个 y 轴来绘制准确率
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:blue')
ax2.plot(accuracies, color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# 显示图形
plt.title("Training Loss and Accuracy")
plt.grid()
plt.show()
