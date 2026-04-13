"""
优化的图片对数据集加载器
避免一次性生成所有图片对，使用动态采样策略
"""
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class ImagePairDataset(Dataset):
    """图片对数据集，用于训练Siamese网络"""
    
    def __init__(self, root_dir, transform=None, samples_per_class=10, positive_ratio=0.5):
        """
        Args:
            root_dir: 数据集根目录，包含多个子文件夹，每个子文件夹代表一个类别
            transform: 图像预处理函数
            samples_per_class: 每个类别每个epoch采样的样本对数量
            positive_ratio: 正样本（同类）的比例
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples_per_class = samples_per_class
        self.positive_ratio = positive_ratio
        
        # 读取所有类别文件夹和图片
        self.class_to_images = {}  # {class_name: [image_paths]}
        self.classes = []
        
        print("正在加载数据集...")
        for class_name in os.listdir(root_dir):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
                
            image_files = [
                os.path.join(class_folder, img_name)
                for img_name in os.listdir(class_folder)
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            if len(image_files) >= 2:  # 至少需要2张图片才能生成正样本对
                self.class_to_images[class_name] = image_files
                self.classes.append(class_name)
        
        print(f"数据集加载完成！共{len(self.classes)}个类别")
        
        # 生成当前epoch的样本对
        self.pairs = []
        self._generate_pairs()
    
    def _generate_pairs(self):
        """动态生成样本对"""
        self.pairs = []
        
        for class_name in self.classes:
            images = self.class_to_images[class_name]
            
            # 生成正样本对（同类）
            num_positive = int(self.samples_per_class * self.positive_ratio)
            for _ in range(num_positive):
                if len(images) >= 2:
                    img1, img2 = random.sample(images, 2)
                    self.pairs.append((img1, img2, 1))
            
            # 生成负样本对（不同类）
            num_negative = self.samples_per_class - num_positive
            for _ in range(num_negative):
                img1 = random.choice(images)
                # 随机选择一个不同的类别
                other_class = random.choice([c for c in self.classes if c != class_name])
                img2 = random.choice(self.class_to_images[other_class])
                self.pairs.append((img1, img2, 0))
        
        # 打乱样本对
        random.shuffle(self.pairs)
        print(f"生成了{len(self.pairs)}对样本（正样本:{sum(1 for p in self.pairs if p[2]==1)}, 负样本:{sum(1 for p in self.pairs if p[2]==0)}）")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path_1, img_path_2, label = self.pairs[idx]
        
        # 加载图片
        try:
            image_1 = Image.open(img_path_1).convert('RGB')
            image_2 = Image.open(img_path_2).convert('RGB')
        except Exception as e:
            print(f"加载图片失败: {img_path_1} 或 {img_path_2}, 错误: {e}")
            # 返回一个默认的黑色图片
            image_1 = Image.new('RGB', (224, 224))
            image_2 = Image.new('RGB', (224, 224))
        
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        
        return image_1, image_2, torch.tensor(label, dtype=torch.float32)
    
    def regenerate_pairs(self):
        """重新生成样本对，用于新的epoch"""
        self._generate_pairs()


class ValidationDataset(Dataset):
    """验证集数据集，使用固定的样本对"""
    
    def __init__(self, root_dir, transform=None, num_pairs_per_class=5):
        """
        Args:
            root_dir: 数据集根目录
            transform: 图像预处理函数
            num_pairs_per_class: 每个类别生成的验证样本对数量
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 读取所有类别文件夹和图片
        self.class_to_images = {}
        self.classes = []
        
        print("正在加载验证集...")
        for class_name in os.listdir(root_dir):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
                
            image_files = [
                os.path.join(class_folder, img_name)
                for img_name in os.listdir(class_folder)
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            if len(image_files) >= 2:
                self.class_to_images[class_name] = image_files
                self.classes.append(class_name)
        
        print(f"验证集加载完成！共{len(self.classes)}个类别")
        
        # 生成固定的验证样本对
        self.pairs = []
        random.seed(42)  # 固定随机种子，确保验证集一致
        
        for class_name in self.classes:
            images = self.class_to_images[class_name]
            
            # 生成正样本对
            for _ in range(num_pairs_per_class):
                if len(images) >= 2:
                    img1, img2 = random.sample(images, 2)
                    self.pairs.append((img1, img2, 1))
            
            # 生成负样本对
            for _ in range(num_pairs_per_class):
                img1 = random.choice(images)
                other_class = random.choice([c for c in self.classes if c != class_name])
                img2 = random.choice(self.class_to_images[other_class])
                self.pairs.append((img1, img2, 0))
        
        random.shuffle(self.pairs)
        print(f"生成了{len(self.pairs)}对验证样本")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path_1, img_path_2, label = self.pairs[idx]
        
        try:
            image_1 = Image.open(img_path_1).convert('RGB')
            image_2 = Image.open(img_path_2).convert('RGB')
        except Exception as e:
            print(f"加载图片失败: {img_path_1} 或 {img_path_2}, 错误: {e}")
            image_1 = Image.new('RGB', (224, 224))
            image_2 = Image.new('RGB', (224, 224))
        
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        
        return image_1, image_2, torch.tensor(label, dtype=torch.float32)

