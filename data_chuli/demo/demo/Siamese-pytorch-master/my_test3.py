import os


def rename_sort(folder_path):

    subfolders = [f for f in os.listdir(fold_path) if os.path.isdir(os.path.join(fold_path, f))]
    subfolders.sort()

    for i, folder_name in enumerate(subfolders):


        old_folder = os.path.join(folder_path, folder_name)
        new_folder = os.path.join(folder_path, f"chapter{i}")

        print(f"{folder_name}已更改为chapter{i}")

        os.rename(old_folder, new_folder)



fold_path = r"E:\photo"
rename_sort(fold_path)

