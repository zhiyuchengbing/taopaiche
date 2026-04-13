"""
测试NaN修复是否成功
运行一个小batch验证训练是否正常
"""
import torch
import torch.nn as nn
from torchvision import transforms
import clip
import sys

sys.path.insert(0, '.')
from train import SiameseCLIP, ContrastiveLoss
from dataset import ImagePairDataset
import config


def test_single_batch():
    """测试单个batch是否会出现nan"""
    print("\n" + "=" * 70)
    print("测试训练是否会出现NaN")
    print("=" * 70)
    
    # 设置设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 加载模型
    print("加载CLIP模型...")
    clip_model, _ = clip.load(config.CLIP_MODEL_NAME, device=device)
    model = SiameseCLIP(
        clip_model,
        freeze_backbone=config.FREEZE_BACKBONE,
        num_layers_to_finetune=config.NUM_LAYERS_TO_FINETUNE
    ).to(device)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    # 创建小数据集
    print("\n加载测试数据...")
    dataset = ImagePairDataset(
        root_dir=config.DATA_DIR,
        transform=transform,
        samples_per_class=2,  # 只用2个样本测试
        positive_ratio=0.5
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(8, len(dataset)),
        shuffle=True
    )
    
    # 损失函数和优化器
    criterion = ContrastiveLoss(margin=config.MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # 测试前向传播
    print("\n" + "=" * 70)
    print("步骤1: 测试前向传播")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (img1, img2, labels) in enumerate(dataloader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  输入形状: img1={img1.shape}, img2={img2.shape}")
            print(f"  标签: {labels.tolist()}")
            
            # 前向传播
            output1, output2 = model(img1, img2)
            
            print(f"  输出形状: output1={output1.shape}, output2={output2.shape}")
            print(f"  输出统计:")
            print(f"    output1 - min: {output1.min():.4f}, max: {output1.max():.4f}, mean: {output1.mean():.4f}")
            print(f"    output2 - min: {output2.min():.4f}, max: {output2.max():.4f}, mean: {output2.mean():.4f}")
            
            # 检查是否有nan或inf
            if output1.isnan().any() or output2.isnan().any():
                print("  ❌ 输出包含NaN!")
                return False
            if output1.isinf().any() or output2.isinf().any():
                print("  ❌ 输出包含Inf!")
                return False
            
            # 计算损失
            loss = criterion(output1, output2, labels)
            print(f"  Loss: {loss.item():.6f}")
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("  ❌ Loss是NaN或Inf!")
                return False
            
            print("  ✓ 前向传播正常")
            break  # 只测试第一个batch
    
    # 测试反向传播
    print("\n" + "=" * 70)
    print("步骤2: 测试反向传播")
    print("=" * 70)
    
    model.train()
    for batch_idx, (img1, img2, labels) in enumerate(dataloader):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
        print(f"\nBatch {batch_idx + 1}:")
        
        # 前向传播
        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, labels)
        
        print(f"  Loss: {loss.item():.6f}")
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        has_nan_grad = False
        grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if param.grad.isnan().any() or param.grad.isinf().any():
                    print(f"  ❌ 参数 {name} 的梯度包含NaN/Inf!")
                    has_nan_grad = True
                grad_norm += param.grad.norm().item() ** 2
        
        grad_norm = grad_norm ** 0.5
        print(f"  梯度范数: {grad_norm:.6f}")
        
        if has_nan_grad:
            return False
        
        # 应用梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        print("  ✓ 反向传播正常")
        break  # 只测试第一个batch
    
    # 测试多个迭代
    print("\n" + "=" * 70)
    print("步骤3: 测试连续训练")
    print("=" * 70)
    
    model.train()
    losses = []
    
    for iteration in range(5):  # 测试5次迭代
        total_loss = 0
        count = 0
        
        for batch_idx, (img1, img2, labels) in enumerate(dataloader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ❌ 迭代 {iteration + 1}, Batch {batch_idx + 1}: Loss是NaN/Inf!")
                return False
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
        
        avg_loss = total_loss / count
        losses.append(avg_loss)
        print(f"  迭代 {iteration + 1}: 平均Loss = {avg_loss:.6f}")
    
    print("\n  Loss趋势:", " -> ".join([f"{l:.4f}" for l in losses]))
    print("  ✓ 连续训练正常")
    
    # 总结
    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)
    print("\n✅ 所有测试通过！没有检测到NaN问题。")
    print("\n建议:")
    print("  1. 可以开始正式训练: python train.py")
    print("  2. 如果训练中仍出现NaN，检查数据集是否有问题")
    print("  3. 可以尝试降低学习率: LEARNING_RATE = 5e-7")
    
    return True


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("CLIP 微调 - NaN修复验证")
    print("=" * 70)
    
    print("\n当前配置:")
    print(f"  模型: {config.CLIP_MODEL_NAME}")
    print(f"  学习率: {config.LEARNING_RATE}")
    print(f"  Batch大小: {config.BATCH_SIZE}")
    print(f"  Margin: {config.MARGIN}")
    print(f"  混合精度: {config.USE_AMP}")
    print(f"  冻结主干: {config.FREEZE_BACKBONE}")
    
    try:
        success = test_single_batch()
        
        if not success:
            print("\n" + "=" * 70)
            print("❌ 测试失败 - 仍存在NaN问题")
            print("=" * 70)
            print("\n建议:")
            print("  1. 降低学习率: LEARNING_RATE = 1e-7")
            print("  2. 减小margin: MARGIN = 0.2")
            print("  3. 使用更小的模型: CLIP_MODEL_NAME = 'RN50'")
            print("  4. 查看详细修复指南: fix_nan_loss.md")
    
    except Exception as e:
        print(f"\n❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n请检查:")
        print("  1. 数据集是否存在: ../output1")
        print("  2. CLIP模型是否正确安装")
        print("  3. CUDA是否可用（如果使用GPU）")


if __name__ == '__main__':
    main()

