import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import font_manager
from model import SimpleCNN

# 设置中文字体
font = font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family'] = font.get_name()

# 加载模型
model = SimpleCNN(input_shape=(32, 32, 3))
model.load()

# 预处理函数
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    resized_img = img.resize((32, 32))
    img_array = np.array(resized_img) / 255.0
    return img_array, img  # 返回预处理数组 + 原图

# 数据路径
cat_dir = 'datasets/cats'
dog_dir = 'datasets/dogs'

# 随机选取一张猫和狗图
random_cat_image_path = random.choice([os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith('.jpg')])
random_dog_image_path = random.choice([os.path.join(dog_dir, f) for f in os.listdir(dog_dir) if f.endswith('.jpg')])

# 图像处理
cat_img_array, cat_original_img = preprocess_image(random_cat_image_path)
dog_img_array, dog_original_img = preprocess_image(random_dog_image_path)

# 模型预测
cat_output = model.forward(cat_img_array.reshape(1, 32, 32, 3))[0]
dog_output = model.forward(dog_img_array.reshape(1, 32, 32, 3))[0]
cat_pred_class = np.argmax(cat_output)
dog_pred_class = np.argmax(dog_output)
cat_prob = cat_output[0]
dog_prob = dog_output[1]

# 模型输入图反归一化显示用
cat_input_img_disp = (cat_img_array * 255).astype(np.uint8)
dog_input_img_disp = (dog_img_array * 255).astype(np.uint8)

# 开始绘图：2行2列
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 第一行：原图
axes[0, 0].imshow(cat_original_img)
axes[0, 0].axis('off')
axes[0, 0].set_title("原图：猫", fontsize=14)

axes[0, 1].imshow(dog_original_img)
axes[0, 1].axis('off')
axes[0, 1].set_title("原图：狗", fontsize=14)

# 第二行：模型输入图 + 预测
axes[1, 0].imshow(cat_input_img_disp)
axes[1, 0].axis('off')
axes[1, 0].set_title(f"预测：{'猫' if cat_pred_class == 0 else '狗'}\n"
                     f"猫概率: {cat_prob:.4f}  狗概率: {1 - cat_prob:.4f}", fontsize=12)

axes[1, 1].imshow(dog_input_img_disp)
axes[1, 1].axis('off')
axes[1, 1].set_title(f"预测：{'猫' if dog_pred_class == 0 else '狗'}\n"
                     f"猫概率: {1 - dog_prob:.4f}  狗概率: {dog_prob:.4f}", fontsize=12)

plt.tight_layout()
plt.show()
