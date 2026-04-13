"""
配置文件
"""
import os

# 路径配置
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'output1')  # 数据集路径
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

# 创建必要的目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 模型配置
CLIP_MODEL_NAME = "ViT-B/32"  # 可选: "ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"

# 微调配置
FREEZE_BACKBONE = True  # 是否冻结主干网络，只微调最后几层
NUM_LAYERS_TO_FINETUNE = 3  # 微调的层数（从后往前数，仅对ViT有效）
# 说明：
# - FREEZE_BACKBONE=True, NUM_LAYERS_TO_FINETUNE=3: 只微调最后3层（推荐，节省显存和训练时间）
# - FREEZE_BACKBONE=True, NUM_LAYERS_TO_FINETUNE=6: 微调最后6层（更灵活，但需要更多显存）
# - FREEZE_BACKBONE=False: 微调所有层（效果可能更好，但训练慢，容易过拟合）

# 训练配置
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 5e-7  # 降低学习率避免NaN（原来是1e-6）
WEIGHT_DECAY = 1e-5   # 降低权重衰减

# 数据集配置
SAMPLES_PER_CLASS = 20  # 每个类别每个epoch采样的样本对数量
POSITIVE_RATIO = 0.5  # 正样本比例
TRAIN_SPLIT = 0.8  # 训练集比例
VAL_PAIRS_PER_CLASS = 5  # 验证集每个类别的样本对数量

# 损失函数配置
MARGIN = 0.5  # Contrastive Loss的margin
TEMPERATURE = 0.07  # Temperature for similarity

# 优化器配置
OPTIMIZER = 'AdamW'  # 可选: 'Adam', 'AdamW', 'SGD'
SCHEDULER = 'CosineAnnealingLR'  # 可选: 'StepLR', 'CosineAnnealingLR', None

# 训练设置
NUM_WORKERS = 4  # DataLoader的工作线程数
SAVE_FREQ = 5  # 每N个epoch保存一次模型
DEVICE = 'cuda'  # 'cuda' or 'cpu'
USE_AMP = False  # 是否使用混合精度训练（设为False避免nan问题）

# 早停配置
EARLY_STOPPING_PATIENCE = 10  # 验证集loss不下降的最大epoch数
MIN_DELTA = 1e-4  # 最小改善阈值

# 图像配置
IMAGE_SIZE = 224  # CLIP输入图像大小
NORMALIZE_MEAN = [0.48145466, 0.4578275, 0.40821073]  # CLIP标准化均值
NORMALIZE_STD = [0.26862954, 0.26130258, 0.27577711]  # CLIP标准化标准差

