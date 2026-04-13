"""
测试冻结层配置
显示模型的可训练参数数量和层数信息
"""
import torch
import clip
import sys

sys.path.insert(0, '.')
from train import SiameseCLIP
import config


def count_parameters(model):
    """统计参数数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def test_freeze_config():
    """测试不同的冻结配置"""
    print("\n" + "=" * 70)
    print("CLIP 冻结层配置测试")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")
    print(f"模型: {config.CLIP_MODEL_NAME}")
    
    # 加载CLIP模型
    print(f"\n加载CLIP模型...")
    clip_model, _ = clip.load(config.CLIP_MODEL_NAME, device=device)
    
    # 检测模型架构
    if hasattr(clip_model.visual, 'transformer'):
        model_type = "ViT"
        total_blocks = len(clip_model.visual.transformer.resblocks)
        print(f"模型类型: Vision Transformer (ViT)")
        print(f"Transformer层数: {total_blocks}")
    elif hasattr(clip_model.visual, 'layer4'):
        model_type = "ResNet"
        print(f"模型类型: ResNet")
    else:
        model_type = "Unknown"
        print(f"模型类型: 未知")
    
    print("\n" + "=" * 70)
    
    # 测试1: 当前配置
    print(f"\n【当前配置】")
    print(f"  FREEZE_BACKBONE = {config.FREEZE_BACKBONE}")
    if config.FREEZE_BACKBONE:
        print(f"  NUM_LAYERS_TO_FINETUNE = {config.NUM_LAYERS_TO_FINETUNE}")
    print("-" * 70)
    
    clip_model_1, _ = clip.load(config.CLIP_MODEL_NAME, device=device)
    model_1 = SiameseCLIP(
        clip_model_1,
        freeze_backbone=config.FREEZE_BACKBONE,
        num_layers_to_finetune=config.NUM_LAYERS_TO_FINETUNE
    )
    
    total_1, trainable_1, frozen_1 = count_parameters(model_1)
    print(f"\n参数统计:")
    print(f"  总参数: {total_1:,}")
    print(f"  可训练: {trainable_1:,} ({100*trainable_1/total_1:.2f}%)")
    print(f"  冻结的: {frozen_1:,} ({100*frozen_1/total_1:.2f}%)")
    
    # 测试2: 对比不同配置
    print("\n" + "=" * 70)
    print("【不同配置对比】")
    print("=" * 70)
    
    configs = []
    
    if model_type == "ViT":
        configs = [
            ("只微调最后3层", True, 3),
            ("只微调最后6层", True, 6),
            ("只微调最后9层", True, 9),
            ("微调所有层", False, 0),
        ]
    else:
        configs = [
            ("只微调ResNet Layer4", True, 0),
            ("微调所有层", False, 0),
        ]
    
    print(f"\n{'配置':<20} {'总参数':<15} {'可训练':<15} {'比例':<10} {'训练速度'}")
    print("-" * 70)
    
    for name, freeze, num_layers in configs:
        clip_model_test, _ = clip.load(config.CLIP_MODEL_NAME, device=device)
        model_test = SiameseCLIP(clip_model_test, freeze_backbone=freeze, num_layers_to_finetune=num_layers)
        total, trainable, frozen = count_parameters(model_test)
        ratio = 100 * trainable / total
        
        # 估算训练速度（相对值）
        if ratio < 10:
            speed = "⭐⭐⭐⭐⭐ 最快"
        elif ratio < 30:
            speed = "⭐⭐⭐⭐   很快"
        elif ratio < 60:
            speed = "⭐⭐⭐     中等"
        else:
            speed = "⭐⭐       较慢"
        
        print(f"{name:<20} {total:>12,}   {trainable:>12,}   {ratio:>6.2f}%   {speed}")
    
    # 测试3: 查看具体哪些层被冻结
    print("\n" + "=" * 70)
    print("【当前配置详细信息】")
    print("=" * 70)
    
    print("\n可训练的参数组:")
    for name, param in model_1.named_parameters():
        if param.requires_grad:
            print(f"  ✓ {name:<60} {param.numel():>10,}")
    
    print("\n冻结的参数组（前10个）:")
    count = 0
    for name, param in model_1.named_parameters():
        if not param.requires_grad:
            print(f"  ❄ {name:<60} {param.numel():>10,}")
            count += 1
            if count >= 10:
                frozen_count = sum(1 for p in model_1.parameters() if not p.requires_grad)
                if frozen_count > 10:
                    print(f"  ... 还有 {frozen_count - 10} 个冻结的参数组")
                break
    
    # 建议
    print("\n" + "=" * 70)
    print("【建议】")
    print("=" * 70)
    
    if trainable_1 / total_1 < 0.1:
        print("\n✓ 当前配置非常节省资源，适合快速训练和显存不足的情况")
        print("  优点: 训练快、显存占用少、不容易过拟合")
        print("  缺点: 模型灵活性有限")
    elif trainable_1 / total_1 < 0.3:
        print("\n✓ 当前配置平衡了速度和效果，适合大多数场景")
        print("  优点: 训练较快、效果较好")
        print("  缺点: 需要中等显存")
    elif trainable_1 / total_1 < 0.6:
        print("\n✓ 当前配置追求更好效果，需要较多资源")
        print("  优点: 模型更灵活、效果可能更好")
        print("  缺点: 训练慢、需要大显存、容易过拟合")
    else:
        print("\n⚠ 当前配置训练所有层，需要大量资源")
        print("  优点: 最大灵活性、理论上效果最好")
        print("  缺点: 训练很慢、需要大量显存和数据、容易过拟合")
        print("\n建议: 如果数据集不是特别大（>50000张），考虑冻结部分层")
    
    print("\n" + "=" * 70)
    print("修改配置: 编辑 config.py 中的 FREEZE_BACKBONE 和 NUM_LAYERS_TO_FINETUNE")
    print("详细说明: 查看 微调说明.md 和 参数配置示例.txt")
    print("=" * 70)


if __name__ == '__main__':
    test_freeze_config()

