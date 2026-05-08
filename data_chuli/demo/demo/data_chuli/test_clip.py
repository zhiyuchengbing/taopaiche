import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import clip
from torchvision import transforms

class ImagePairDataset(Dataset):
    def __init__(self, root_dir, clip_model, transform=None):
        self.root_dir = root_dir
        self.clip_model = clip_model
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 读取所有类别文件夹
        self.classes = os.listdir(root_dir)
        
        for label, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            # 读取每个文件夹中的图片
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

        # 创建正负样本对
        self.pairs = []
        for i, img_path_1 in enumerate(self.image_paths):
            for j, img_path_2 in enumerate(self.image_paths):
                # 如果是同一类
                if self.labels[i] == self.labels[j]:
                    self.pairs.append((img_path_1, img_path_2, 1))
                # 如果是不同类
                else:
                    self.pairs.append((img_path_1, img_path_2, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path_1, img_path_2, label = self.pairs[idx]
        
        # 加载图片
        image_1 = Image.open(img_path_1)
        image_2 = Image.open(img_path_2)
        
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        
        return image_1, image_2, label

# 设置 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device)

# 数据增强（例如，标准化、裁剪等）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

# 创建数据集
dataset = ImagePairDataset("/path/to/dataset", clip_model, transform)
