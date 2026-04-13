"""
模型预测脚本 - 判断两张图片是否属于同一类
"""
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import clip
import argparse

import config
from train import SiameseCLIP


class ImageSimilarityPredictor:
    """图片相似度预测器"""
    
    def __init__(self, checkpoint_path=None, threshold=None):
        """
        Args:
            checkpoint_path: 模型权重路径
            threshold: 距离阈值
        """
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold if threshold is not None else config.MARGIN
        
        # 加载CLIP模型
        print(f"加载CLIP模型: {config.CLIP_MODEL_NAME}")
        clip_model, _ = clip.load(config.CLIP_MODEL_NAME, device=self.device)
        
        # 创建模型（推理时不需要冻结参数）
        self.model = SiameseCLIP(clip_model, freeze_backbone=False).to(self.device)
        
        # 加载权重
        if checkpoint_path is None:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到模型文件: {checkpoint_path}")
        
        print(f"加载模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"模型加载成功! (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f})")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
        ])
    
    def load_image(self, image_path):
        """加载并预处理图片"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)  # 添加batch维度
            return image_tensor
        except Exception as e:
            raise ValueError(f"加载图片失败 {image_path}: {e}")
    
    def predict(self, image1_path, image2_path, return_distance=False):
        """
        预测两张图片是否属于同一类
        
        Args:
            image1_path: 第一张图片路径
            image2_path: 第二张图片路径
            return_distance: 是否返回距离值
        
        Returns:
            如果return_distance=True: (is_same_class, distance, confidence)
            否则: is_same_class
        """
        # 加载图片
        img1 = self.load_image(image1_path).to(self.device)
        img2 = self.load_image(image2_path).to(self.device)
        
        # 预测
        with torch.no_grad():
            output1, output2 = self.model(img1, img2)
            
            # 计算距离
            distance = nn.functional.pairwise_distance(output1, output2).item()
            
            # 判断是否同类
            is_same_class = distance < self.threshold
            
            # 计算置信度（距离越小，置信度越高）
            if is_same_class:
                confidence = 1 - (distance / self.threshold)  # 同类时，距离越小置信度越高
            else:
                confidence = (distance - self.threshold) / (1 - self.threshold)  # 不同类时，距离越大置信度越高
                confidence = min(confidence, 1.0)
        
        if return_distance:
            return is_same_class, distance, confidence
        else:
            return is_same_class
    
    def predict_batch(self, image_pairs):
        """
        批量预测
        
        Args:
            image_pairs: 图片对列表 [(img1_path, img2_path), ...]
        
        Returns:
            结果列表 [(is_same_class, distance, confidence), ...]
        """
        results = []
        for img1_path, img2_path in image_pairs:
            result = self.predict(img1_path, img2_path, return_distance=True)
            results.append(result)
        return results
    
    def find_similar_images(self, query_image_path, candidate_folder, top_k=5):
        """
        在候选文件夹中找到与查询图片最相似的k张图片
        
        Args:
            query_image_path: 查询图片路径
            candidate_folder: 候选图片文件夹
            top_k: 返回前k个最相似的图片
        
        Returns:
            [(image_path, distance, is_same_class), ...]
        """
        # 获取候选图片
        candidate_images = [
            os.path.join(candidate_folder, f)
            for f in os.listdir(candidate_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        # 计算距离
        results = []
        for candidate_path in candidate_images:
            try:
                is_same, distance, _ = self.predict(query_image_path, candidate_path, return_distance=True)
                results.append((candidate_path, distance, is_same))
            except Exception as e:
                print(f"处理图片失败 {candidate_path}: {e}")
        
        # 按距离排序
        results.sort(key=lambda x: x[1])
        
        return results[:top_k]


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='CLIP图片相似度预测')
    parser.add_argument('--img1', type=str, required=True, help='第一张图片路径')
    parser.add_argument('--img2', type=str, required=True, help='第二张图片路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型权重路径')
    parser.add_argument('--threshold', type=float, default=None, help='距离阈值')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = ImageSimilarityPredictor(
        checkpoint_path=args.checkpoint,
        threshold=args.threshold
    )
    
    # 预测
    print("\n" + "=" * 50)
    print(f"图片1: {args.img1}")
    print(f"图片2: {args.img2}")
    print("=" * 50)
    
    is_same, distance, confidence = predictor.predict(
        args.img1,
        args.img2,
        return_distance=True
    )
    
    print(f"\n预测结果:")
    print(f"  是否同类: {'是' if is_same else '否'}")
    print(f"  欧氏距离: {distance:.4f}")
    print(f"  置信度: {confidence * 100:.2f}%")
    print(f"  阈值: {predictor.threshold:.4f}")
    
    if is_same:
        print(f"\n✓ 这两张图片属于同一类（距离 {distance:.4f} < 阈值 {predictor.threshold:.4f}）")
    else:
        print(f"\n✗ 这两张图片不属于同一类（距离 {distance:.4f} >= 阈值 {predictor.threshold:.4f}）")


if __name__ == '__main__':
    main()

