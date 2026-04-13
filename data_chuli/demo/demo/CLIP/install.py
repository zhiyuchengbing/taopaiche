"""
自动安装脚本
帮助用户快速安装所有依赖
"""
import subprocess
import sys
import os


def print_banner():
    """打印欢迎信息"""
    print("\n" + "=" * 60)
    print("CLIP 微调系统 - 自动安装脚本")
    print("=" * 60)


def check_python_version():
    """检查Python版本"""
    print("\n检查Python版本...")
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("✗ Python版本过低，需要3.7或更高版本")
        return False
    
    print("✓ Python版本符合要求")
    return True


def check_cuda():
    """检查CUDA是否可用"""
    print("\n检查CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA可用")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠ CUDA不可用，将使用CPU模式（训练速度较慢）")
            return False
    except ImportError:
        print("⚠ PyTorch未安装，稍后将安装")
        return False


def install_pytorch():
    """安装PyTorch"""
    print("\n安装PyTorch...")
    print("请选择安装方式:")
    print("1. CPU版本（适用于没有NVIDIA GPU的电脑）")
    print("2. CUDA 11.8版本（适用于NVIDIA GPU）")
    print("3. CUDA 12.1版本（适用于较新的NVIDIA GPU）")
    print("4. 跳过（已安装）")
    
    choice = input("\n请输入选项 (1-4): ")
    
    if choice == '1':
        cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"]
    elif choice == '2':
        cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118"]
    elif choice == '3':
        cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"]
    elif choice == '4':
        print("跳过PyTorch安装")
        return True
    else:
        print("无效选项，跳过")
        return False
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ PyTorch安装成功")
        return True
    except subprocess.CalledProcessError:
        print("✗ PyTorch安装失败")
        return False


def install_clip():
    """安装CLIP"""
    print("\n安装CLIP...")
    
    # 先安装基础依赖
    base_deps = ["ftfy", "regex", "tqdm"]
    for dep in base_deps:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
        except subprocess.CalledProcessError:
            print(f"⚠ 安装 {dep} 失败")
    
    # 安装CLIP
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"], check=True)
        print("✓ CLIP安装成功")
        return True
    except subprocess.CalledProcessError:
        print("✗ CLIP安装失败")
        print("\n请尝试手动安装:")
        print("  git clone https://github.com/openai/CLIP.git")
        print("  cd CLIP")
        print("  pip install -e .")
        return False


def install_other_dependencies():
    """安装其他依赖"""
    print("\n安装其他依赖...")
    
    deps = [
        "Pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "seaborn>=0.11.0",
        "numpy>=1.19.0"
    ]
    
    for dep in deps:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✓ 安装 {dep.split('>=')[0]}")
        except subprocess.CalledProcessError:
            print(f"✗ 安装 {dep.split('>=')[0]} 失败")


def verify_installation():
    """验证安装"""
    print("\n验证安装...")
    
    # 测试导入
    packages = ['torch', 'torchvision', 'clip', 'PIL', 'matplotlib', 'sklearn', 'seaborn', 'numpy']
    all_ok = True
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} 导入失败")
            all_ok = False
    
    return all_ok


def create_directories():
    """创建必要的目录"""
    print("\n创建目录...")
    
    dirs = ['checkpoints', 'logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ {d}/")


def show_next_steps():
    """显示下一步操作"""
    print("\n" + "=" * 60)
    print("安装完成!")
    print("=" * 60)
    print("\n下一步操作:")
    print("1. 确保数据集在 ../output1 目录下")
    print("2. 运行快速开始: python quick_start.py")
    print("3. 或者直接训练: python train.py")
    print("4. 查看详细文档: README.md 和 使用说明.md")
    print("\n" + "=" * 60)


def main():
    """主函数"""
    print_banner()
    
    # 检查Python版本
    if not check_python_version():
        return
    
    # 安装PyTorch
    print("\n" + "=" * 60)
    print("步骤1: 安装PyTorch")
    print("=" * 60)
    install_pytorch()
    
    # 检查CUDA
    check_cuda()
    
    # 安装CLIP
    print("\n" + "=" * 60)
    print("步骤2: 安装CLIP")
    print("=" * 60)
    install_clip()
    
    # 安装其他依赖
    print("\n" + "=" * 60)
    print("步骤3: 安装其他依赖")
    print("=" * 60)
    install_other_dependencies()
    
    # 验证安装
    print("\n" + "=" * 60)
    print("步骤4: 验证安装")
    print("=" * 60)
    if verify_installation():
        print("\n✓ 所有包安装成功!")
    else:
        print("\n⚠ 部分包安装失败，请检查错误信息")
    
    # 创建目录
    create_directories()
    
    # 显示下一步
    show_next_steps()


if __name__ == '__main__':
    main()

