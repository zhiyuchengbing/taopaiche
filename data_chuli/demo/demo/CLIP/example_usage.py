"""
使用示例脚本
展示如何使用训练好的模型进行预测
"""
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from predict import ImageSimilarityPredictor


def example_single_prediction():
    """示例1: 单张图片对比"""
    print("=" * 60)
    print("示例1: 单张图片对比")
    print("=" * 60)
    
    # 创建预测器
    try:
        predictor = ImageSimilarityPredictor()
    except FileNotFoundError:
        print("错误: 找不到训练好的模型")
        print("请先运行 python train.py 训练模型")
        return
    
    # 这里替换成你的图片路径
    img1_path = "../output1/晋KA4977/20230508093308_tile3.jpg"
    img2_path = "../output1/晋KA4977/20230508093308_tile3_aug_dark.jpg"
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"警告: 示例图片不存在，请修改图片路径")
        return
    
    # 预测
    is_same, distance, confidence = predictor.predict(
        img1_path,
        img2_path,
        return_distance=True
    )
    
    print(f"\n图片1: {img1_path}")
    print(f"图片2: {img2_path}")
    print(f"\n结果:")
    print(f"  是否同类: {'是 ✓' if is_same else '否 ✗'}")
    print(f"  欧氏距离: {distance:.4f}")
    print(f"  置信度: {confidence * 100:.2f}%")


def example_batch_prediction():
    """示例2: 批量预测"""
    print("\n" + "=" * 60)
    print("示例2: 批量预测")
    print("=" * 60)
    
    try:
        predictor = ImageSimilarityPredictor()
    except FileNotFoundError:
        print("错误: 找不到训练好的模型")
        return
    
    # 准备图片对列表（请根据实际情况修改）
    image_pairs = [
        ("../output1/晋KA4977/20230508093308_tile3.jpg", 
         "../output1/晋KA4977/20230508093308_tile3_aug_dark.jpg"),  # 应该是同类
        ("../output1/晋KA4977/20230508093308_tile3.jpg", 
         "../output1/川ADP799/20230728101208_tile3.jpg"),  # 应该是不同类
    ]
    
    # 检查图片是否存在
    valid_pairs = []
    for img1, img2 in image_pairs:
        if os.path.exists(img1) and os.path.exists(img2):
            valid_pairs.append((img1, img2))
    
    if not valid_pairs:
        print("警告: 没有找到有效的图片对")
        return
    
    # 批量预测
    results = predictor.predict_batch(valid_pairs)
    
    print(f"\n共预测 {len(results)} 对图片:\n")
    for i, ((img1, img2), (is_same, distance, confidence)) in enumerate(zip(valid_pairs, results)):
        print(f"图片对 {i+1}:")
        print(f"  图片1: {os.path.basename(img1)}")
        print(f"  图片2: {os.path.basename(img2)}")
        print(f"  结果: {'同类 ✓' if is_same else '不同类 ✗'}")
        print(f"  距离: {distance:.4f}")
        print(f"  置信度: {confidence * 100:.2f}%")
        print()


def example_find_similar():
    """示例3: 查找相似图片"""
    print("=" * 60)
    print("示例3: 查找相似图片")
    print("=" * 60)
    
    try:
        predictor = ImageSimilarityPredictor()
    except FileNotFoundError:
        print("错误: 找不到训练好的模型")
        return
    
    # 查询图片
    query_image = "../output1/晋KA4977/20230508093308_tile3.jpg"
    
    if not os.path.exists(query_image):
        print("警告: 查询图片不存在")
        return
    
    # 在某个文件夹中查找相似图片
    candidate_folder = "../output1/晋KA4977"
    
    if not os.path.exists(candidate_folder):
        print("警告: 候选文件夹不存在")
        return
    
    print(f"\n查询图片: {query_image}")
    print(f"在文件夹 {candidate_folder} 中查找最相似的图片...\n")
    
    # 查找前5个最相似的
    similar_images = predictor.find_similar_images(
        query_image_path=query_image,
        candidate_folder=candidate_folder,
        top_k=5
    )
    
    print(f"找到 {len(similar_images)} 张相似图片:\n")
    for i, (img_path, distance, is_same) in enumerate(similar_images):
        print(f"{i+1}. {os.path.basename(img_path)}")
        print(f"   距离: {distance:.4f}")
        print(f"   {'同类 ✓' if is_same else '不同类 ✗'}")
        print()


def example_compare_different_classes():
    """示例4: 比较不同类别的图片"""
    print("=" * 60)
    print("示例4: 比较不同类别的车辆")
    print("=" * 60)
    
    try:
        predictor = ImageSimilarityPredictor()
    except FileNotFoundError:
        print("错误: 找不到训练好的模型")
        return
    
    # 获取数据集中的一些类别
    data_dir = "../output1"
    if not os.path.exists(data_dir):
        print(f"警告: 数据目录不存在: {data_dir}")
        return
    
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))][:3]
    
    if len(classes) < 2:
        print("警告: 数据集中类别太少")
        return
    
    print(f"\n从以下类别中选择图片进行比较: {classes}\n")
    
    # 从每个类别中选择一张图片
    test_images = []
    for class_name in classes:
        class_folder = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            test_images.append((class_name, os.path.join(class_folder, images[0])))
    
    # 比较所有图片对
    print("比较结果:\n")
    for i in range(len(test_images)):
        for j in range(i + 1, len(test_images)):
            class1, img1 = test_images[i]
            class2, img2 = test_images[j]
            
            is_same, distance, confidence = predictor.predict(img1, img2, return_distance=True)
            
            print(f"{class1} vs {class2}")
            print(f"  预测: {'同类 ✓' if is_same else '不同类 ✗'}")
            print(f"  实际: {'同类' if class1 == class2 else '不同类'}")
            print(f"  距离: {distance:.4f}")
            print(f"  置信度: {confidence * 100:.2f}%")
            
            # 判断预测是否正确
            is_correct = is_same == (class1 == class2)
            print(f"  {'预测正确 ✓' if is_correct else '预测错误 ✗'}")
            print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("CLIP模型使用示例")
    print("=" * 60)
    
    # 示例1: 单张图片对比
    example_single_prediction()
    
    # 示例2: 批量预测
    example_batch_prediction()
    
    # 示例3: 查找相似图片
    example_find_similar()
    
    # 示例4: 比较不同类别
    example_compare_different_classes()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()

