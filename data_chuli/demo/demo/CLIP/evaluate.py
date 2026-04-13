"""
模型评估脚本
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import clip
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import ValidationDataset
import config
from train import SiameseCLIP


def evaluate_model(model, dataloader, device, threshold=None):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        threshold: 距离阈值，如果为None则使用config.MARGIN
    """
    if threshold is None:
        threshold = config.MARGIN
    
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_distances = []
    
    print("正在评估模型...")
    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            # 前向传播
            output1, output2 = model(img1, img2)
            
            # 计算距离
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            
            # 预测
            predictions = (euclidean_distance < threshold).float()
            
            # 收集结果
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_distances.extend(euclidean_distance.cpu().numpy())
    
    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_distances = np.array(all_distances)
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"准确率 (Accuracy): {accuracy * 100:.2f}%")
    print(f"精确率 (Precision): {precision * 100:.2f}%")
    print(f"召回率 (Recall): {recall * 100:.2f}%")
    print(f"F1分数 (F1-Score): {f1 * 100:.2f}%")
    print(f"\n混淆矩阵:")
    print(cm)
    print(f"\nTN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    
    # 统计距离分布
    same_class_distances = all_distances[all_labels == 1]
    diff_class_distances = all_distances[all_labels == 0]
    
    print(f"\n距离统计:")
    print(f"同类图片平均距离: {same_class_distances.mean():.4f} ± {same_class_distances.std():.4f}")
    print(f"不同类图片平均距离: {diff_class_distances.mean():.4f} ± {diff_class_distances.std():.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'same_class_distances': same_class_distances,
        'diff_class_distances': diff_class_distances,
        'all_labels': all_labels,
        'all_predictions': all_predictions,
        'all_distances': all_distances
    }


def plot_confusion_matrix(cm, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()
    print(f"混淆矩阵已保存到: {save_path}")


def plot_distance_distribution(same_class_distances, diff_class_distances, threshold, save_path):
    """绘制距离分布图"""
    plt.figure(figsize=(10, 6))
    plt.hist(same_class_distances, bins=50, alpha=0.5, label='Same Class', color='green')
    plt.hist(diff_class_distances, bins=50, alpha=0.5, label='Different Class', color='red')
    plt.axvline(x=threshold, color='blue', linestyle='--', label=f'Threshold={threshold:.2f}')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"距离分布图已保存到: {save_path}")


def find_optimal_threshold(all_labels, all_distances):
    """寻找最优阈值"""
    thresholds = np.linspace(all_distances.min(), all_distances.max(), 100)
    best_threshold = 0
    best_f1 = 0
    
    for threshold in thresholds:
        predictions = (all_distances < threshold).astype(int)
        f1 = f1_score(all_labels, predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n最优阈值: {best_threshold:.4f} (F1-Score: {best_f1 * 100:.2f}%)")
    return best_threshold


def main():
    """主评估函数"""
    print("=" * 50)
    print("CLIP模型评估")
    print("=" * 50)
    
    # 设置设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载CLIP模型
    print(f"加载CLIP模型: {config.CLIP_MODEL_NAME}")
    clip_model, _ = clip.load(config.CLIP_MODEL_NAME, device=device)
    
    # 创建模型（推理时不需要冻结参数）
    model = SiameseCLIP(clip_model, freeze_backbone=False).to(device)
    
    # 加载训练好的权重
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到模型文件 {checkpoint_path}")
        print("请先运行train.py训练模型")
        return
    
    print(f"加载模型权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载的模型来自 Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    # 创建验证集
    print("\n加载验证集...")
    val_dataset = ValidationDataset(
        root_dir=config.DATA_DIR,
        transform=transform,
        num_pairs_per_class=config.VAL_PAIRS_PER_CLASS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # 评估模型
    results = evaluate_model(model, val_loader, device)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        results['confusion_matrix'],
        os.path.join(config.LOG_DIR, 'confusion_matrix.png')
    )
    
    # 绘制距离分布
    plot_distance_distribution(
        results['same_class_distances'],
        results['diff_class_distances'],
        config.MARGIN,
        os.path.join(config.LOG_DIR, 'distance_distribution.png')
    )
    
    # 寻找最优阈值
    optimal_threshold = find_optimal_threshold(results['all_labels'], results['all_distances'])
    
    # 使用最优阈值重新评估
    print("\n" + "=" * 50)
    print(f"使用最优阈值 {optimal_threshold:.4f} 重新评估")
    print("=" * 50)
    results_optimal = evaluate_model(model, val_loader, device, threshold=optimal_threshold)
    
    # 保存结果
    with open(os.path.join(config.LOG_DIR, 'evaluation_results.txt'), 'w', encoding='utf-8') as f:
        f.write("评估结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"使用阈值: {config.MARGIN}\n")
        f.write(f"准确率: {results['accuracy'] * 100:.2f}%\n")
        f.write(f"精确率: {results['precision'] * 100:.2f}%\n")
        f.write(f"召回率: {results['recall'] * 100:.2f}%\n")
        f.write(f"F1分数: {results['f1'] * 100:.2f}%\n\n")
        
        f.write(f"最优阈值: {optimal_threshold:.4f}\n")
        f.write(f"最优准确率: {results_optimal['accuracy'] * 100:.2f}%\n")
        f.write(f"最优精确率: {results_optimal['precision'] * 100:.2f}%\n")
        f.write(f"最优召回率: {results_optimal['recall'] * 100:.2f}%\n")
        f.write(f"最优F1分数: {results_optimal['f1'] * 100:.2f}%\n")
    
    print(f"\n评估结果已保存到: {config.LOG_DIR}")
    print("评估完成!")


if __name__ == '__main__':
    main()

