"""
货车套牌识别数据增强脚本
功能：模拟不同光照和天气条件，增强数据集
作者：AI Assistant
日期：2025-10-27
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
import glob

class TruckDataAugmentor:
    """货车数据增强类"""
    
    def __init__(self, input_dir, min_images=5):
        """
        初始化
        Args:
            input_dir: 输入文件夹路径
            min_images: 每个类别最少图片数量
        """
        self.input_dir = input_dir
        self.min_images = min_images
        self.stats = {
            'total_folders': 0,
            'processed_folders': 0,
            'total_original': 0,
            'total_augmented': 0,
            'skipped_folders': 0,
            'corrupted_files': 0
        }
        self.corrupted_files_list = []
    
    def augment_brightness_up(self, image):
        """增强1: 白天强光"""
        # 转换为PIL格式
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 亮度增强
        brightness = ImageEnhance.Brightness(img_pil)
        img_pil = brightness.enhance(np.random.uniform(1.3, 1.5))
        
        # 对比度增强
        contrast = ImageEnhance.Contrast(img_pil)
        img_pil = contrast.enhance(1.2)
        
        # 饱和度增强
        color = ImageEnhance.Color(img_pil)
        img_pil = color.enhance(1.1)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def augment_brightness_down(self, image):
        """增强2: 夜间弱光"""
        # 降低亮度
        img = image.astype(np.float32)
        brightness_factor = np.random.uniform(0.4, 0.6)
        img = img * brightness_factor
        
        # 添加高斯噪声
        noise = np.random.normal(0, 5, image.shape)
        img = img + noise
        
        # 降低对比度
        img = img * 0.9
        
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def augment_rainy(self, image):
        """增强3: 阴雨天"""
        # 运动模糊（模拟雨滴）
        kernel_size = 3
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        img = cv2.filter2D(image, -1, kernel)
        
        # 降低亮度
        img = img.astype(np.float32)
        img = img * np.random.uniform(0.75, 0.85)
        
        # 增加蓝色色调
        img[:, :, 0] = np.clip(img[:, :, 0] + 10, 0, 255)  # B通道
        
        # 降低饱和度
        img_pil = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        color = ImageEnhance.Color(img_pil)
        img_pil = color.enhance(0.85)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def augment_foggy(self, image):
        """增强4: 雾天/霾天"""
        img = image.astype(np.float32)
        
        # 降低对比度
        img_pil = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        contrast = ImageEnhance.Contrast(img_pil)
        img_pil = contrast.enhance(np.random.uniform(0.6, 0.75))
        
        # 降低饱和度
        color = ImageEnhance.Color(img_pil)
        img_pil = color.enhance(0.7)
        
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR).astype(np.float32)
        
        # 叠加白雾
        fog = np.ones_like(img) * 255
        alpha = np.random.uniform(0.3, 0.5)
        img = cv2.addWeighted(img.astype(np.uint8), 1-alpha, fog.astype(np.uint8), alpha, 0)
        
        return img
    
    def augment_overexposed(self, image):
        """增强5: 曝光过度"""
        img = image.astype(np.float32) / 255.0
        
        # 伽马校正
        gamma = np.random.uniform(1.3, 1.5)
        img = np.power(img, gamma)
        
        img = (img * 255).astype(np.uint8)
        
        # 增加亮度
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        brightness = ImageEnhance.Brightness(img_pil)
        img_pil = brightness.enhance(np.random.uniform(1.4, 1.6))
        
        # 降低饱和度
        color = ImageEnhance.Color(img_pil)
        img_pil = color.enhance(0.8)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def augment_single_image(self, image_path, output_folder):
        """对单张图片进行增强"""
        # 使用支持中文路径的方式读取图片
        try:
            # 方法1: 使用numpy和cv2.imdecode处理中文路径
            img_array = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                # 方法2: 使用PIL读取再转换
                from PIL import Image as PILImage
                pil_img = PILImage.open(image_path)
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"⚠️ 无法读取图片: {os.path.basename(image_path)} - {str(e)[:50]}")
            self.corrupted_files_list.append(image_path)
            self.stats['corrupted_files'] += 1
            return 0
        
        if image is None:
            print(f"⚠️ 图片损坏或格式不支持: {os.path.basename(image_path)}")
            self.corrupted_files_list.append(image_path)
            self.stats['corrupted_files'] += 1
            return 0
        
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ext = os.path.splitext(image_path)[1]
        
        augmented_count = 0
        
        # 应用各种增强
        augmentations = {
            'bright': self.augment_brightness_up,
            'dark': self.augment_brightness_down,
            'rainy': self.augment_rainy,
            'foggy': self.augment_foggy,
            'overexp': self.augment_overexposed
        }
        
        for aug_name, aug_func in augmentations.items():
            try:
                augmented_img = aug_func(image.copy())
                output_path = os.path.join(output_folder, f"{base_name}_aug_{aug_name}{ext}")
                
                # 检查文件是否已存在
                if not os.path.exists(output_path):
                    # 使用支持中文路径的方式保存图片
                    _, img_encode = cv2.imencode(ext, augmented_img)
                    img_encode.tofile(output_path)
                    augmented_count += 1
            except Exception as e:
                print(f"⚠️ 增强失败 {aug_name} for {os.path.basename(image_path)}: {str(e)[:50]}")
        
        return augmented_count
    
    def process_folder(self, folder_path):
        """处理单个车牌文件夹"""
        folder_name = os.path.basename(folder_path)
        
        # 获取所有原始图片（不包括已增强的）
        all_images = glob.glob(os.path.join(folder_path, '*.[jJ][pP][gG]')) + \
                     glob.glob(os.path.join(folder_path, '*.[jJ][pP][eE][gG]')) + \
                     glob.glob(os.path.join(folder_path, '*.[pP][nN][gG]'))
        
        # 过滤掉已经增强的图片
        original_images = [img for img in all_images if '_aug_' not in os.path.basename(img)]
        
        if not original_images:
            return False
        
        original_count = len(original_images)
        current_total = len(all_images)
        
        # 如果已经达到最小数量，跳过
        if current_total >= self.min_images:
            return False
        
        # 计算需要增强的图片数量
        need_augment = max(0, self.min_images - current_total)
        
        if need_augment == 0:
            return False
        
        # 对原始图片进行增强
        augmented_total = 0
        for img_path in original_images:
            count = self.augment_single_image(img_path, folder_path)
            augmented_total += count
            
            # 检查是否已经达到目标
            if current_total + augmented_total >= self.min_images:
                break
        
        self.stats['total_original'] += original_count
        self.stats['total_augmented'] += augmented_total
        
        return True
    
    def run(self):
        """执行数据增强"""
        print("=" * 70)
        print("🚛 货车套牌识别数据增强系统")
        print("=" * 70)
        print(f"📁 输入目录: {self.input_dir}")
        print(f"🎯 目标: 每个类别至少 {self.min_images} 张图片")
        print(f"🎨 增强策略: 白天强光、夜间弱光、阴雨天、雾天、曝光过度")
        print("=" * 70)
        
        # 获取所有车牌文件夹
        folders = [f for f in glob.glob(os.path.join(self.input_dir, '*')) if os.path.isdir(f)]
        self.stats['total_folders'] = len(folders)
        
        print(f"\n📊 找到 {len(folders)} 个车牌类别文件夹\n")
        
        # 处理每个文件夹
        with tqdm(folders, desc="处理进度", unit="文件夹") as pbar:
            for folder in pbar:
                folder_name = os.path.basename(folder)
                pbar.set_description(f"处理: {folder_name[:15]}")
                
                if self.process_folder(folder):
                    self.stats['processed_folders'] += 1
                else:
                    self.stats['skipped_folders'] += 1
        
        # 打印统计信息
        self.print_statistics()
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 70)
        print("📊 数据增强完成统计")
        print("=" * 70)
        print(f"总文件夹数:       {self.stats['total_folders']}")
        print(f"处理的文件夹数:   {self.stats['processed_folders']}")
        print(f"跳过的文件夹数:   {self.stats['skipped_folders']}")
        print(f"原始图片总数:     {self.stats['total_original']}")
        print(f"新增增强图片数:   {self.stats['total_augmented']}")
        print(f"损坏/无法读取:    {self.stats['corrupted_files']}")
        print(f"总图片数:         {self.stats['total_original'] + self.stats['total_augmented']}")
        print("=" * 70)
        
        # 如果有损坏文件，保存列表
        if self.corrupted_files_list:
            corrupted_log = os.path.join(self.input_dir, "corrupted_files.txt")
            with open(corrupted_log, 'w', encoding='utf-8') as f:
                f.write("以下文件损坏或无法读取:\n")
                f.write("=" * 70 + "\n")
                for file_path in self.corrupted_files_list:
                    f.write(f"{file_path}\n")
            print(f"⚠️ 发现 {len(self.corrupted_files_list)} 个损坏文件")
            print(f"📝 损坏文件列表已保存到: {corrupted_log}")
            print("=" * 70)
        
        print("✅ 数据增强完成！")
        print("=" * 70)


def main():
    """主函数"""
    # 配置参数
    INPUT_DIR = r"F:\汽车衡数据集\output1"  # 输入文件夹
    MIN_IMAGES = 5  # 每个类别最少图片数
    
    # 检查输入目录是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误: 输入目录不存在: {INPUT_DIR}")
        return
    
    # 创建增强器并运行
    augmentor = TruckDataAugmentor(INPUT_DIR, MIN_IMAGES)
    augmentor.run()


if __name__ == "__main__":
    main()

