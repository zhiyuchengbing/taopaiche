# -*- coding: utf-8 -*-
"""plate_char_det — 尾部视角车挂号/放大号字符检测+比对模块.

提供 CharReader 类，封装方案B 的两阶段字符检测管线：
  1. yolo_det 检测车挂号(窄框 cls=0)/放大号(宽框 cls=1)
  2. 字符检测器找框 → 49类分类器读字符 → 读序规范化
  3. 方案B 比对 (compare) → 一致/不一致/无法判断/作废

模型权重引用 D:\data2\weibu_zifu 下的成品，不复制到本工程。
"""

from .char_reader import CharReader

__all__ = ["CharReader"]
