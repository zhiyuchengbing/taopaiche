# CLIP模型微调 - 图片相似度判断

本项目使用CLIP模型进行微调，用于判断两张图片是否属于同一类（套牌车识别）。

## 项目结构

```
CLIP/
├── dataset.py          # 数据集加载器
├── config.py           # 配置文件
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
├── predict.py          # 预测脚本
├── requirements.txt    # 依赖包
├── README.md           # 说明文档
├── checkpoints/        # 模型权重保存目录
└── logs/               # 训练日志和可视化结果
```

## 环境安装

### 1. 创建虚拟环境（推荐）

```bash
conda create -n clip python=3.8
conda activate clip
```

### 2. 安装依赖

```bash
cd CLIP
pip install -r requirements.txt
```

**注意**: 如果安装CLIP失败，可以尝试：
```bash
pip install git+https://github.com/openai/CLIP.git
```

或者从源码安装：
```bash
git clone https://github.com/openai/CLIP.git
cd CLIP
pip install -e .
```

## 数据集准备

数据集应该按照以下格式组织：

```
output1/
├── 车牌号1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 车牌号2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

每个文件夹代表一个类别（车牌号），文件夹内的图片属于同一辆车。

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练过程中会：
- 自动加载 `output1` 文件夹中的数据
- 每个epoch动态生成图片对（避免内存溢出）
- 使用对比损失（Contrastive Loss）训练
- 每5个epoch保存一次检查点
- 自动保存最佳模型
- 生成训练曲线和日志

**训练参数配置**（在 `config.py` 中修改）：
- `BATCH_SIZE`: 批次大小（默认32）
- `NUM_EPOCHS`: 训练轮数（默认50）
- `LEARNING_RATE`: 学习率（默认1e-6）
- `SAMPLES_PER_CLASS`: 每个类别每个epoch的样本对数量（默认20）
- `MARGIN`: Contrastive Loss的margin（默认0.5）

### 2. 评估模型

```bash
python evaluate.py
```

评估会输出：
- 准确率、精确率、召回率、F1分数
- 混淆矩阵
- 距离分布图
- 最优阈值建议

### 3. 预测（判断两张图片是否同类）

#### 命令行方式

```bash
python predict.py --img1 path/to/image1.jpg --img2 path/to/image2.jpg
```

可选参数：
- `--checkpoint`: 指定模型权重路径（默认使用最佳模型）
- `--threshold`: 指定距离阈值（默认使用配置文件中的值）

#### Python API方式

```python
from predict import ImageSimilarityPredictor

# 创建预测器
predictor = ImageSimilarityPredictor()

# 预测两张图片是否同类
is_same, distance, confidence = predictor.predict(
    'path/to/image1.jpg',
    'path/to/image2.jpg',
    return_distance=True
)

print(f"是否同类: {is_same}")
print(f"距离: {distance:.4f}")
print(f"置信度: {confidence * 100:.2f}%")

# 批量预测
image_pairs = [
    ('img1.jpg', 'img2.jpg'),
    ('img3.jpg', 'img4.jpg'),
]
results = predictor.predict_batch(image_pairs)

# 找到最相似的图片
similar_images = predictor.find_similar_images(
    query_image_path='query.jpg',
    candidate_folder='candidates/',
    top_k=5
)
```

## 模型说明

### 网络架构

本项目使用**孪生网络（Siamese Network）**架构：
- 基础模型: CLIP ViT-B/32
- 两张图片通过相同的CLIP编码器提取特征
- 计算特征向量之间的欧氏距离
- 使用对比损失进行训练

### 损失函数

**对比损失（Contrastive Loss）**:

```
Loss = y * d^2 + (1-y) * max(margin - d, 0)^2
```

其中：
- `y=1` 表示同类，`y=0` 表示不同类
- `d` 是特征向量之间的欧氏距离
- `margin` 是边界参数

### 判断标准

两张图片是否属于同一类的判断依据：
- 计算特征向量的欧氏距离 `d`
- 如果 `d < threshold`，判定为同类
- 如果 `d >= threshold`，判定为不同类

默认阈值为0.5，可以通过评估脚本找到最优阈值。

## 配置说明

在 `config.py` 中可以修改以下配置：

### 模型配置
- `CLIP_MODEL_NAME`: CLIP模型类型（"ViT-B/32", "ViT-B/16", "ViT-L/14"等）

### 训练配置
- `BATCH_SIZE`: 批次大小
- `NUM_EPOCHS`: 训练轮数
- `LEARNING_RATE`: 学习率
- `WEIGHT_DECAY`: 权重衰减
- `OPTIMIZER`: 优化器类型（Adam, AdamW, SGD）
- `SCHEDULER`: 学习率调度器

### 数据配置
- `SAMPLES_PER_CLASS`: 每个类别每个epoch的样本对数量
- `POSITIVE_RATIO`: 正样本（同类）比例
- `VAL_PAIRS_PER_CLASS`: 验证集每个类别的样本对数量

### 损失配置
- `MARGIN`: Contrastive Loss的margin参数

### 早停配置
- `EARLY_STOPPING_PATIENCE`: 验证集不改善的最大epoch数
- `MIN_DELTA`: 最小改善阈值

## 训练技巧

### 1. 数据增强
训练时使用了以下数据增强：
- 随机水平翻转
- 颜色抖动（亮度和对比度）

### 2. 学习率调度
使用余弦退火学习率调度器，训练更稳定。

### 3. 动态样本生成
每个epoch重新生成图片对，增加数据多样性，避免过拟合。

### 4. 早停机制
当验证集loss连续N个epoch不下降时，自动停止训练。

## 常见问题

### Q1: 训练时显存不足怎么办？
- 减小 `BATCH_SIZE`
- 减小 `SAMPLES_PER_CLASS`
- 使用更小的CLIP模型（如RN50）

### Q2: 如何提高模型性能？
- 增加训练数据（更多类别和图片）
- 调整 `MARGIN` 参数
- 增加数据增强
- 使用更大的CLIP模型（如ViT-L/14）
- 调整 `SAMPLES_PER_CLASS` 和 `POSITIVE_RATIO`

### Q3: 如何选择最优阈值？
运行 `evaluate.py`，脚本会自动计算并推荐最优阈值。

### Q4: 训练需要多长时间？
取决于数据集大小和硬件配置：
- 小数据集（几千张图片）: 几小时
- 中等数据集（1-2万张）: 几小时到一天
- 大数据集（更多）: 可能需要更长时间

### Q5: 可以使用预训练的CLIP模型直接预测吗？
可以，但效果可能不如微调后的模型。CLIP是在通用图像-文本对上预训练的，对特定任务（如车牌识别）需要微调以获得更好的效果。

## 性能优化建议

1. **GPU加速**: 确保使用GPU训练（设置 `DEVICE='cuda'`）
2. **多进程数据加载**: 调整 `NUM_WORKERS` 参数
3. **混合精度训练**: 代码中已使用 `torch.cuda.amp`
4. **批处理预测**: 使用 `predict_batch` 进行批量预测

## 引用

如果使用了CLIP模型，请引用：

```bibtex
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International Conference on Machine Learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}
```

## 许可证

本项目遵循MIT许可证。

## 联系方式

如有问题或建议，欢迎提Issue或PR。

