"""
快速开始脚本 - 一键运行训练和评估
"""
import os
import sys
import subprocess


def check_dependencies():
    """检查依赖是否安装"""
    print("检查依赖包...")
    try:
        import torch
        import clip
        from PIL import Image
        print("✓ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("\n请先安装依赖:")
        print("  pip install -r requirements.txt")
        return False


def check_data():
    """检查数据集"""
    print("\n检查数据集...")
    data_dir = "../output1"
    
    if not os.path.exists(data_dir):
        print(f"✗ 数据集目录不存在: {data_dir}")
        return False
    
    # 统计类别数量
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(classes) == 0:
        print(f"✗ 数据集为空")
        return False
    
    # 统计图片数量
    total_images = 0
    for class_name in classes:
        class_folder = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
    
    print(f"✓ 数据集加载成功")
    print(f"  类别数量: {len(classes)}")
    print(f"  图片数量: {total_images}")
    print(f"  平均每类: {total_images / len(classes):.1f} 张")
    
    return True


def train_model():
    """训练模型"""
    print("\n" + "=" * 60)
    print("开始训练模型...")
    print("=" * 60)
    
    try:
        subprocess.run([sys.executable, "train.py"], check=True)
        print("\n✓ 训练完成")
        return True
    except subprocess.CalledProcessError:
        print("\n✗ 训练失败")
        return False


def evaluate_model():
    """评估模型"""
    print("\n" + "=" * 60)
    print("评估模型...")
    print("=" * 60)
    
    try:
        subprocess.run([sys.executable, "evaluate.py"], check=True)
        print("\n✓ 评估完成")
        return True
    except subprocess.CalledProcessError:
        print("\n✗ 评估失败")
        return False


def show_results():
    """显示结果"""
    print("\n" + "=" * 60)
    print("训练结果")
    print("=" * 60)
    
    # 检查是否有结果文件
    log_dir = "logs"
    checkpoint_dir = "checkpoints"
    
    if os.path.exists(os.path.join(checkpoint_dir, "best_model.pth")):
        print(f"✓ 模型已保存: {checkpoint_dir}/best_model.pth")
    
    if os.path.exists(os.path.join(log_dir, "training_history.txt")):
        print(f"✓ 训练历史: {log_dir}/training_history.txt")
    
    if os.path.exists(os.path.join(log_dir, "evaluation_results.txt")):
        print(f"✓ 评估结果: {log_dir}/evaluation_results.txt")
        # 读取并显示评估结果
        with open(os.path.join(log_dir, "evaluation_results.txt"), 'r', encoding='utf-8') as f:
            print("\n" + f.read())
    
    if os.path.exists(os.path.join(log_dir, "loss_curve.png")):
        print(f"✓ 损失曲线: {log_dir}/loss_curve.png")
    
    if os.path.exists(os.path.join(log_dir, "confusion_matrix.png")):
        print(f"✓ 混淆矩阵: {log_dir}/confusion_matrix.png")
    
    print("\n下一步:")
    print("  1. 查看训练曲线和评估结果")
    print("  2. 使用 predict.py 进行预测")
    print("  3. 查看 example_usage.py 了解更多使用方法")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("CLIP 微调 - 快速开始")
    print("=" * 60)
    
    # 步骤1: 检查依赖
    if not check_dependencies():
        return
    
    # 步骤2: 检查数据
    if not check_data():
        return
    
    # 询问是否开始训练
    print("\n准备就绪!")
    response = input("\n是否开始训练? (y/n): ")
    
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 步骤3: 训练模型
    if not train_model():
        return
    
    # 步骤4: 评估模型
    response = input("\n是否评估模型? (y/n): ")
    if response.lower() == 'y':
        evaluate_model()
    
    # 步骤5: 显示结果
    show_results()
    
    print("\n" + "=" * 60)
    print("快速开始完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()

