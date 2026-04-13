import os

def rename_files_in_folder(folder_path, prefix="chapter"):
    # 获取文件夹中的所有文件列表
    files = os.listdir(folder_path)

    # 对文件按字母或数字顺序排序（可以根据需要修改排序方式）
    files.sort()

    # 遍历文件并重命名
    for i, filename in enumerate(files):
        # 获取文件的扩展名
        file_extension = os.path.splitext(filename)[1]

        # 生成新的文件名
        new_name = f"{prefix}{i}{file_extension}"

        # 获取旧文件的完整路径
        old_file = os.path.join(folder_path, filename)

        # 获取新文件的完整路径
        new_file = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_name}'")

# 使用示例：指定文件夹路径
folder_path = "E:\photo"
rename_files_in_folder(folder_path)
