"""
清理损坏文件工具
功能：删除数据增强过程中发现的损坏文件
作者：AI Assistant
日期：2025-10-27
"""

import os

def clean_corrupted_files(corrupted_list_file):
    """
    根据损坏文件列表删除文件
    
    Args:
        corrupted_list_file: 损坏文件列表的路径
    """
    if not os.path.exists(corrupted_list_file):
        print(f"❌ 错误: 找不到损坏文件列表: {corrupted_list_file}")
        return
    
    # 读取损坏文件列表
    with open(corrupted_list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 跳过标题行
    file_paths = [line.strip() for line in lines if line.strip() and not line.startswith('=') and not line.startswith('以下')]
    
    print("=" * 70)
    print("🗑️ 损坏文件清理工具")
    print("=" * 70)
    print(f"发现 {len(file_paths)} 个损坏文件")
    print()
    
    # 确认删除
    response = input("是否删除这些文件？(yes/no): ")
    if response.lower() not in ['yes', 'y', '是']:
        print("❌ 已取消删除操作")
        return
    
    # 删除文件
    deleted_count = 0
    not_found_count = 0
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_count += 1
                print(f"✅ 已删除: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"⚠️ 删除失败: {os.path.basename(file_path)} - {str(e)}")
        else:
            not_found_count += 1
            print(f"⚠️ 文件不存在: {os.path.basename(file_path)}")
    
    print()
    print("=" * 70)
    print("📊 清理统计")
    print("=" * 70)
    print(f"总文件数:     {len(file_paths)}")
    print(f"成功删除:     {deleted_count}")
    print(f"文件不存在:   {not_found_count}")
    print(f"删除失败:     {len(file_paths) - deleted_count - not_found_count}")
    print("=" * 70)
    print("✅ 清理完成！")
    print("=" * 70)


def main():
    """主函数"""
    corrupted_list = r"F:\汽车衡数据集\output1\corrupted_files.txt"
    
    if not os.path.exists(corrupted_list):
        print(f"❌ 找不到损坏文件列表")
        print(f"请先运行 data_augmentation.py 生成损坏文件列表")
        return
    
    clean_corrupted_files(corrupted_list)


if __name__ == "__main__":
    main()

