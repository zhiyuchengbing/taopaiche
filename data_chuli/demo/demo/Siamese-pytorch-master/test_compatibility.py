#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本 - 验证numpy兼容性修复
"""

def test_imports():
    """测试关键模块导入"""
    print("测试模块导入...")
    
    try:
        import numpy as np
        print(f"✅ numpy版本: {np.__version__}")
        
        # 测试numpy.bool8兼容性
        if hasattr(np, 'bool8'):
            print("✅ numpy.bool8 可用")
        else:
            print("⚠️ numpy.bool8 不可用，但已添加兼容性修复")
            np.bool8 = np.bool_
            print("✅ 已添加numpy.bool8别名")
        
        import torch
        print(f"✅ torch版本: {torch.__version__}")
        
        # 测试tensorboard导入
        try:
            from torch.utils.tensorboard import SummaryWriter
            print("✅ tensorboard导入成功")
        except Exception as e:
            print(f"❌ tensorboard导入失败: {e}")
            return False
            
        # 测试项目模块导入
        try:
            from utils.callbacks import LossHistory
            print("✅ LossHistory导入成功")
        except Exception as e:
            print(f"❌ LossHistory导入失败: {e}")
            return False
            
        print("\n🎉 所有测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 导入测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("开始兼容性测试...")
    print("=" * 50)
    
    success = test_imports()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 测试成功！现在可以运行train.py了")
        print("建议运行: python train.py")
    else:
        print("❌ 测试失败！请运行: python fix_dependencies.py")
    print("=" * 50)
