"""
CLIP模型微调训练脚本
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import clip
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from dataset import ImagePairDataset, ValidationDataset
import config


class ContrastiveLoss(nn.Module):
    """对比损失函数（改进版，数值更稳定）"""
    
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-8
    
    def forward(self, output1, output2, label):
        """
        Args:
            output1: 第一张图片的特征向量 [batch_size, feature_dim]
            output2: 第二张图片的特征向量 [batch_size, feature_dim]
            label: 标签，1表示同类，0表示不同类 [batch_size]
        """
        # 计算欧氏距离，添加eps避免数值问题
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, eps=self.eps)
        
        # 裁剪距离值，避免过大
        euclidean_distance = torch.clamp(euclidean_distance, min=self.eps, max=10.0)
        
        # 对比损失
        pos_loss = label * torch.pow(euclidean_distance, 2)
        neg_loss = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        loss_contrastive = torch.mean(pos_loss + neg_loss)
        
        # 确保loss不会太大
        loss_contrastive = torch.clamp(loss_contrastive, max=10.0)
        
        return loss_contrastive


class SiameseCLIP(nn.Module):
    """基于CLIP的孪生网络"""
    
    def __init__(self, clip_model, freeze_backbone=True, num_layers_to_finetune=3):
        """
        Args:
            clip_model: CLIP模型
            freeze_backbone: 是否冻结主干网络
            num_layers_to_finetune: 微调的层数（从后往前数）
        """
        super(SiameseCLIP, self).__init__()
        self.clip_model = clip_model
        
        if freeze_backbone:
            # 先冻结所有参数
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            # 检测CLIP模型类型并解冻最后几层
            if hasattr(self.clip_model.visual, 'transformer'):
                # ViT架构 (ViT-B/32, ViT-B/16, ViT-L/14等)
                total_blocks = len(self.clip_model.visual.transformer.resblocks)
                layers_to_train = self.clip_model.visual.transformer.resblocks[-num_layers_to_finetune:]
                
                for layer in layers_to_train:
                    for param in layer.parameters():
                        param.requires_grad = True
                
                # 解冻投影层
                if hasattr(self.clip_model.visual, 'proj'):
                    if self.clip_model.visual.proj is not None:
                        self.clip_model.visual.proj.requires_grad = True
                
                print(f"✓ 冻结策略: ViT模型，冻结前{total_blocks - num_layers_to_finetune}层，微调后{num_layers_to_finetune}层")
                
            elif hasattr(self.clip_model.visual, 'layer4'):
                # ResNet架构 (RN50, RN101等)
                # 微调layer4（最后一个残差块）
                for param in self.clip_model.visual.layer4.parameters():
                    param.requires_grad = True
                
                # 解冻投影层
                if hasattr(self.clip_model.visual, 'attnpool'):
                    for param in self.clip_model.visual.attnpool.parameters():
                        param.requires_grad = True
                
                print(f"✓ 冻结策略: ResNet模型，冻结layer1-3，微调layer4和注意力池化层")
            
            # 统计可训练参数
            trainable_params = sum(p.numel() for p in self.clip_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.clip_model.parameters())
            print(f"✓ 可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        else:
            print("✓ 微调所有层")
    
    def forward_one(self, x):
        """前向传播单张图片"""
        features = self.clip_model.encode_image(x)
        
        # L2归一化，添加eps避免除零
        norm = features.norm(dim=-1, keepdim=True)
        # 避免norm太小导致数值不稳定
        norm = torch.clamp(norm, min=1e-7)
        features = features / norm
        
        # 确保特征值在合理范围内
        features = torch.clamp(features, min=-10.0, max=10.0)
        
        return features.float()  # 确保输出是float32
    
    def forward(self, input1, input2):
        """前向传播图片对"""
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (img1, img2, labels) in enumerate(pbar):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        
        if scaler is not None:
            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, labels)
            
            # 检查loss是否为nan
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: Loss is {loss.item()}, skipping batch")
                continue
            
            # 反向传播
            scaler.scale(loss).backward()
            # 梯度裁剪（更严格）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 不使用混合精度
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels)
            
            # 检查loss是否为nan
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: Loss is {loss.item()}, skipping batch")
                continue
            
            # 反向传播
            loss.backward()
            # 梯度裁剪（更严格）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
        
        # 检查参数是否有nan
        for name, param in model.named_parameters():
            if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
                print(f"\nWarning: Parameter {name} contains NaN/Inf after update!")
                # 重置优化器状态
                optimizer.zero_grad()
                continue
        
        # 统计
        running_loss += loss.item()
        
        # 计算准确率（使用距离阈值判断）
        with torch.no_grad():
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            predictions = (euclidean_distance < config.MARGIN).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        # 更新进度条
        if total > 0:
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader, desc='Validation'):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            # 前向传播
            output1, output2 = model(img1, img2)
            
            # 计算损失
            loss = criterion(output1, output2, labels)
            running_loss += loss.item()
            
            # 计算准确率
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            predictions = (euclidean_distance < config.MARGIN).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def plot_losses(train_losses, val_losses, save_path):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main():
    """主训练函数"""
    print("=" * 50)
    print("CLIP微调训练")
    print("=" * 50)
    
    # 设置设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载CLIP模型
    print(f"加载CLIP模型: {config.CLIP_MODEL_NAME}")
    clip_model, preprocess = clip.load(config.CLIP_MODEL_NAME, device=device)
    
    # 创建孪生网络
    print(f"\n模型配置:")
    print(f"  冻结主干: {config.FREEZE_BACKBONE}")
    if config.FREEZE_BACKBONE:
        print(f"  微调层数: {config.NUM_LAYERS_TO_FINETUNE}")
    
    model = SiameseCLIP(
        clip_model,
        freeze_backbone=config.FREEZE_BACKBONE,
        num_layers_to_finetune=config.NUM_LAYERS_TO_FINETUNE
    ).to(device)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),  # 数据增强
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    # 创建数据集
    print("\n加载数据集...")
    train_dataset = ImagePairDataset(
        root_dir=config.DATA_DIR,
        transform=transform,
        samples_per_class=config.SAMPLES_PER_CLASS,
        positive_ratio=config.POSITIVE_RATIO
    )
    
    val_dataset = ValidationDataset(
        root_dir=config.DATA_DIR,
        transform=val_transform,
        num_pairs_per_class=config.VAL_PAIRS_PER_CLASS
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # 损失函数
    criterion = ContrastiveLoss(margin=config.MARGIN)
    
    print(f"\n损失函数:")
    print(f"  类型: Contrastive Loss")
    print(f"  Margin: {config.MARGIN}")
    
    # 优化器
    if config.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
    
    # 学习率调度器
    if config.SCHEDULER == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif config.SCHEDULER == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    else:
        scheduler = None
    
    # 混合精度训练
    scaler = None
    if config.USE_AMP and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("✓ 启用混合精度训练")
    else:
        print("✓ 使用FP32训练（更稳定）")
    
    # 训练历史
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 训练循环
    print("\n开始训练...")
    print("=" * 50)
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # 重新生成训练样本对（增加数据多样性）
        if epoch > 1:
            train_dataset.regenerate_pairs()
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=config.NUM_WORKERS,
                pin_memory=True
            )
        
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 打印结果
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 保存最佳模型
        if val_loss < best_val_loss - config.MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
            print(f"保存最佳模型! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        # 定期保存检查点
        if epoch % config.SAVE_FREQ == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'))
        
        # 早停
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n早停触发! 验证集loss已经{patience_counter}个epoch没有改善")
            break
        
        print("-" * 50)
    
    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
    }, os.path.join(config.CHECKPOINT_DIR, 'final_model.pth'))
    
    # 绘制损失曲线
    plot_losses(train_losses, val_losses, os.path.join(config.LOG_DIR, 'loss_curve.png'))
    
    # 保存训练历史
    with open(os.path.join(config.LOG_DIR, 'training_history.txt'), 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Train_Acc,Val_Acc\n")
        for i in range(len(train_losses)):
            f.write(f"{i+1},{train_losses[i]:.4f},{val_losses[i]:.4f},{train_accs[i]:.2f},{val_accs[i]:.2f}\n")
    
    print("\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存路径: {config.CHECKPOINT_DIR}")
    print(f"日志保存路径: {config.LOG_DIR}")


if __name__ == '__main__':
    main()

