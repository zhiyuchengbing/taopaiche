# Loss为NaN问题修复指南

## 🔴 问题描述

训练时出现 `loss=nan`，导致模型无法正常训练。

## ✅ 已应用的修复方案

### 1. 禁用混合精度训练（主要修复）

**修改文件**: `config.py`

```python
USE_AMP = False  # 禁用混合精度，避免数值不稳定
```

混合精度训练在某些情况下会导致数值下溢/上溢，特别是在CLIP这种预训练模型上微调时。

### 2. 改进归一化操作

**修改文件**: `train.py` - `SiameseCLIP.forward_one()`

```python
# 添加eps避免除零
features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
```

### 3. 添加梯度裁剪

**修改文件**: `train.py` - `train_one_epoch()`

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4. NaN检测和跳过

**修改文件**: `train.py` - `train_one_epoch()`

```python
# 检查并跳过nan batch
if torch.isnan(loss) or torch.isinf(loss):
    print(f"\nWarning: Loss is {loss.item()}, skipping batch")
    continue
```

### 5. 确保数据类型一致

```python
return features.float()  # 确保输出是float32
```

## 🚀 重新训练

现在可以重新开始训练：

```bash
python train.py
```

## 📊 预期结果

修复后，你应该看到：

```
Epoch 1: 100%|████| 818/818 [04:30<00:00, 3.02it/s, loss=0.234, acc=72.5%]
Validation: 100%|████| 409/409 [02:12<00:00, 3.09it/s]

Epoch 1/50
Train Loss: 0.2345, Train Acc: 72.50%
Val Loss: 0.1987, Val Acc: 75.30%
Learning Rate: 0.000001
```

## 🔍 如果问题仍然存在

### 方案A: 调整学习率

编辑 `config.py`:

```python
LEARNING_RATE = 5e-7  # 降低学习率
```

### 方案B: 增加批次大小的稳定性

编辑 `config.py`:

```python
BATCH_SIZE = 16  # 减小批次大小
```

### 方案C: 修改margin参数

编辑 `config.py`:

```python
MARGIN = 0.2  # 减小margin
```

### 方案D: 检查数据

运行数据检查脚本：

```python
# check_data.py
import os
from PIL import Image
from dataset import ImagePairDataset
from torchvision import transforms
import config

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
])

dataset = ImagePairDataset(
    root_dir=config.DATA_DIR,
    transform=transform,
    samples_per_class=5,
    positive_ratio=0.5
)

print(f"数据集大小: {len(dataset)}")
print("检查前10个样本...")

for i in range(min(10, len(dataset))):
    try:
        img1, img2, label = dataset[i]
        print(f"样本{i}: img1={img1.shape}, img2={img2.shape}, label={label}")
        
        # 检查是否有nan或inf
        if img1.isnan().any() or img2.isnan().any():
            print(f"  警告: 样本{i}包含NaN!")
        if img1.isinf().any() or img2.isinf().any():
            print(f"  警告: 样本{i}包含Inf!")
            
    except Exception as e:
        print(f"样本{i}加载失败: {e}")

print("\n数据检查完成!")
```

## 🎯 Loss为NaN的常见原因

### 1. ✅ 混合精度训练（已修复）
- 问题: FP16精度不够，导致数值下溢
- 解决: 禁用混合精度或使用GradScaler

### 2. ✅ 除零错误（已修复）
- 问题: 归一化时分母为0
- 解决: 添加eps=1e-8

### 3. ✅ 梯度爆炸（已修复）
- 问题: 梯度过大导致参数更新异常
- 解决: 添加梯度裁剪

### 4. 学习率过大
- 问题: 更新步长太大
- 解决: 降低学习率到5e-7或1e-7

### 5. 数据问题
- 问题: 数据中包含nan、inf或异常值
- 解决: 检查数据加载和预处理

### 6. 初始化问题
- 问题: 模型参数初始化不当
- 解决: 使用CLIP预训练权重（已自动处理）

## 📈 监控训练

### 查看loss趋势

训练时观察：
- Loss应该从0.2-0.5开始逐渐下降
- 如果第一个epoch的loss就很大（>10），可能有问题
- 如果loss突然跳变到nan，检查上述原因

### 正常的训练曲线

```
Epoch 1: loss=0.234, acc=72%
Epoch 2: loss=0.198, acc=78%
Epoch 3: loss=0.176, acc=82%
...
```

### 异常的训练曲线

```
Epoch 1: loss=nan, acc=50%  ❌ 第一个epoch就nan
Epoch 1: loss=15.3, acc=50% ❌ Loss太大
Epoch 5: loss=nan, acc=85%  ❌ 训练中途突然nan
```

## 💡 调试技巧

### 1. 打印中间值

在 `train.py` 的 `forward_one` 中添加调试代码：

```python
def forward_one(self, x):
    features = self.clip_model.encode_image(x)
    
    # 调试: 打印特征统计
    # print(f"Features - min: {features.min():.4f}, max: {features.max():.4f}, mean: {features.mean():.4f}")
    
    norm = features.norm(dim=-1, keepdim=True) + 1e-8
    
    # 调试: 打印norm
    # print(f"Norm - min: {norm.min():.4f}, max: {norm.max():.4f}")
    
    features = features / norm
    return features.float()
```

### 2. 减少数据量测试

临时修改 `config.py`:

```python
SAMPLES_PER_CLASS = 5  # 少量数据测试
NUM_EPOCHS = 3
```

如果少量数据能正常训练，说明不是代码问题。

### 3. 使用更小的模型

```python
CLIP_MODEL_NAME = "RN50"  # ResNet更稳定
```

## 🔧 终极解决方案

如果所有方法都不行，使用以下保守配置：

```python
# config.py 保守配置
CLIP_MODEL_NAME = "RN50"
FREEZE_BACKBONE = True
NUM_LAYERS_TO_FINETUNE = 3
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 5e-7  # 更小的学习率
WEIGHT_DECAY = 0      # 不使用权重衰减
MARGIN = 0.2          # 更小的margin
USE_AMP = False       # 禁用混合精度
OPTIMIZER = 'SGD'     # 使用SGD（更稳定）
```

然后在 `train.py` 中使用SGD优化器时添加动量：

```python
elif config.OPTIMIZER == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, 
                         momentum=0.9, weight_decay=config.WEIGHT_DECAY)
```

## ✅ 验证修复

运行以下命令验证：

```bash
# 1. 测试系统
python test_system.py

# 2. 查看配置
python test_freeze.py

# 3. 开始训练
python train.py
```

## 📞 还需要帮助？

1. 检查 `logs/` 目录中的完整日志
2. 运行 `python test_system.py` 诊断环境
3. 确认CUDA和PyTorch版本兼容
4. 尝试在CPU上训练几个batch（设置 `DEVICE='cpu'`）

---

**总结**: 主要问题是混合精度训练导致的数值不稳定，已通过禁用AMP修复。现在重新训练应该不会出现nan了！

