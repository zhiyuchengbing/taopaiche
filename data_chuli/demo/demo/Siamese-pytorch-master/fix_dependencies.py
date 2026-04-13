#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
依赖修复脚本
解决numpy版本兼容性问题
"""

import subprocess
import sys
import os

def run_command(command):
    """运行命令并显示输出"""
    print(f"执行命令: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True, encoding='utf-8')
        print(result.stdout)
        if result.stderr:
            print("警告:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        print(f"输出: {e.stdout}")
        print(f"错误信息: {e.stderr}")
        return False

def fix_numpy_compatibility():
    """修复numpy兼容性问题"""
    print("=" * 50)
    print("开始修复numpy版本兼容性问题...")
    print("=" * 50)
    
    # 方案1: 升级tensorboard到兼容版本
    print("\n方案1: 升级tensorboard...")
    if run_command("pip install tensorboard>=2.8.0 --upgrade"):
        print("✅ tensorboard升级成功")
    else:
        print("❌ tensorboard升级失败")
    
    # 方案2: 安装兼容的numpy版本
    print("\n方案2: 安装兼容的numpy版本...")
    if run_command("pip install 'numpy>=1.19.0,<1.24.0' --upgrade"):
        print("✅ numpy版本调整成功")
    else:
        print("❌ numpy版本调整失败")
    
    # 方案3: 重新安装所有依赖
    print("\n方案3: 重新安装项目依赖...")
    if os.path.exists("requirements.txt"):
        if run_command("pip install -r requirements.txt --upgrade"):
            print("✅ 依赖重新安装成功")
        else:
            print("❌ 依赖重新安装失败")
    
    print("\n" + "=" * 50)
    print("修复完成！请尝试重新运行您的脚本。")
    print("=" * 50)

if __name__ == "__main__":
    fix_numpy_compatibility()
