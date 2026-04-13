"""
系统测试脚本
验证CLIP微调系统是否正常工作
"""
import os
import sys


def test_imports():
    """测试所有必需的包是否可以导入"""
    print("\n" + "=" * 60)
    print("测试1: 检查依赖包")
    print("=" * 60)
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'clip': 'CLIP',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'seaborn': 'Seaborn',
        'numpy': 'NumPy',
        'tqdm': 'TQDM'
    }
    
    failed = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:20s} - 已安装")
        except ImportError:
            print(f"✗ {name:20s} - 未安装")
            failed.append(name)
    
    if failed:
        print(f"\n缺少以下包: {', '.join(failed)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("\n✓ 所有依赖包已安装")
    return True


def test_cuda():
    """测试CUDA是否可用"""
    print("\n" + "=" * 60)
    print("测试2: 检查CUDA")
    print("=" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用")
            print(f"  版本: {torch.version.cuda}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    显存: {props.total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠ CUDA 不可用，将使用CPU模式")
            print("  训练速度会较慢")
            return True  # CPU模式也是可以的
    except Exception as e:
        print(f"✗ 检查CUDA时出错: {e}")
        return False


def test_clip_model():
    """测试CLIP模型加载"""
    print("\n" + "=" * 60)
    print("测试3: 加载CLIP模型")
    print("=" * 60)
    
    try:
        import torch
        import clip
        
        print("正在加载 ViT-B/32 模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        print(f"✓ CLIP模型加载成功")
        print(f"  设备: {device}")
        print(f"  模型类型: ViT-B/32")
        
        # 测试模型输入
        import numpy as np
        from PIL import Image
        
        # 创建一个测试图片
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        image_input = preprocess(test_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        
        print(f"  特征维度: {image_features.shape}")
        print("✓ CLIP模型工作正常")
        
        return True
    except Exception as e:
        print(f"✗ 加载CLIP模型失败: {e}")
        print("\n可能的解决方法:")
        print("1. 确保网络连接正常（首次运行需要下载模型）")
        print("2. 尝试手动安装CLIP:")
        print("   pip install git+https://github.com/openai/CLIP.git")
        return False


def test_dataset():
    """测试数据集"""
    print("\n" + "=" * 60)
    print("测试4: 检查数据集")
    print("=" * 60)
    
    try:
        import config
        
        data_dir = config.DATA_DIR
        print(f"数据集路径: {data_dir}")
        
        if not os.path.exists(data_dir):
            print(f"✗ 数据集目录不存在: {data_dir}")
            print("\n请确保数据集在正确的位置")
            return False
        
        # 统计类别
        classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        if len(classes) == 0:
            print(f"✗ 数据集为空")
            return False
        
        # 统计图片
        total_images = 0
        valid_classes = 0
        for class_name in classes:
            class_folder = os.path.join(data_dir, class_name)
            images = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) >= 2:  # 至少需要2张图片
                total_images += len(images)
                valid_classes += 1
        
        print(f"✓ 数据集检查完成")
        print(f"  总类别数: {len(classes)}")
        print(f"  有效类别: {valid_classes} (每类至少2张图)")
        print(f"  总图片数: {total_images}")
        print(f"  平均每类: {total_images / valid_classes:.1f} 张")
        
        if valid_classes < 2:
            print("\n⚠ 警告: 有效类别太少，建议至少有10个类别")
            return True
        
        if total_images < 100:
            print("\n⚠ 警告: 图片数量较少，建议至少有1000张图片")
            return True
        
        return True
        
    except Exception as e:
        print(f"✗ 检查数据集时出错: {e}")
        return False


def test_modules():
    """测试项目模块"""
    print("\n" + "=" * 60)
    print("测试5: 检查项目模块")
    print("=" * 60)
    
    modules = {
        'config': '配置文件',
        'dataset': '数据集加载器',
        'train': '训练脚本',
        'evaluate': '评估脚本',
        'predict': '预测脚本'
    }
    
    failed = []
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✓ {name:20s} - 正常")
        except Exception as e:
            print(f"✗ {name:20s} - 错误: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n以下模块有问题: {', '.join(failed)}")
        return False
    
    print("\n✓ 所有项目模块正常")
    return True


def test_directories():
    """测试必要的目录"""
    print("\n" + "=" * 60)
    print("测试6: 检查目录结构")
    print("=" * 60)
    
    try:
        import config
        
        dirs = {
            config.CHECKPOINT_DIR: '模型保存目录',
            config.LOG_DIR: '日志目录'
        }
        
        for dir_path, name in dirs.items():
            if os.path.exists(dir_path):
                print(f"✓ {name:20s} - 存在: {dir_path}")
            else:
                os.makedirs(dir_path, exist_ok=True)
                print(f"✓ {name:20s} - 已创建: {dir_path}")
        
        return True
    except Exception as e:
        print(f"✗ 检查目录时出错: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("CLIP微调系统 - 系统测试")
    print("=" * 60)
    
    tests = [
        ("依赖包", test_imports),
        ("CUDA", test_cuda),
        ("CLIP模型", test_clip_model),
        ("数据集", test_dataset),
        ("项目模块", test_modules),
        ("目录结构", test_directories)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n测试 {test_name} 时发生异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name:20s}: {status}")
    
    print(f"\n通过率: {passed}/{total} ({passed*100/total:.0f}%)")
    
    if passed == total:
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！系统可以正常使用")
        print("=" * 60)
        print("\n下一步:")
        print("1. 运行训练: python train.py")
        print("2. 或使用快速开始: python quick_start.py")
    else:
        print("\n" + "=" * 60)
        print("⚠ 部分测试未通过，请检查错误信息")
        print("=" * 60)
        print("\n建议:")
        print("1. 检查依赖是否完全安装: pip install -r requirements.txt")
        print("2. 确保数据集在正确位置")
        print("3. 查看使用说明.md了解详细信息")


if __name__ == '__main__':
    main()

