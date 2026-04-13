#!/usr/bin/env python3
"""
修复历史复检记录中的中文编码问题
将AI结果从中文转换为英文
"""

import os
import json
import datetime
from typing import Dict, Any

def fix_history_ai_recheck():
    """修复历史AI复检记录中的中文编码问题"""
    
    # 设置路径
    stats_logs_dir = "stats_logs"
    
    # 中文到英文的映射
    result_mapping = {
        "套牌": "fake_plate",
        "换挂": "change_trailer", 
        "正常": "normal",
        "无法判断": "unknown"
    }
    
    fixed_count = 0
    error_count = 0
    
    # 遍历所有日期目录
    if not os.path.exists(stats_logs_dir):
        print(f"stats_logs目录不存在: {stats_logs_dir}")
        return
    
    for date_folder in os.listdir(stats_logs_dir):
        date_path = os.path.join(stats_logs_dir, date_folder)
        if not os.path.isdir(date_path):
            continue
            
        print(f"处理日期目录: {date_folder}")
        
        # 处理该日期的JSONL文件
        jsonl_file = os.path.join(stats_logs_dir, f"stats_{date_folder}.jsonl")
        if not os.path.exists(jsonl_file):
            continue
            
        try:
            # 读取所有行
            lines = []
            file_updated = False
            
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        updated = False
                        
                        # 检查是否有AI复检信息
                        ai_recheck = record.get("ai_recheck", {})
                        if ai_recheck.get("attempted", False):
                            ai_result = ai_recheck.get("ai_result", "")
                            
                            # 如果AI结果是中文，转换为英文
                            if ai_result in result_mapping:
                                english_result = result_mapping[ai_result]
                                record["ai_recheck"]["ai_result"] = english_result
                                
                                # 如果复检成功，同时更新case_type
                                if ai_recheck.get("success", False) and english_result in ["fake_plate", "change_trailer", "normal"]:
                                    old_case_type = record.get("case_type", "")
                                    record["case_type"] = english_result
                                    
                                    print(f"  记录 {record.get('record_id', 'unknown')}: "
                                          f"case_type {old_case_type} -> {english_result}, "
                                          f"AI结果 {ai_result} -> {english_result}")
                                
                                updated = True
                                fixed_count += 1
                        
                        # 如果记录有更新，保存修改后的行
                        if updated:
                            lines.append(json.dumps(record, ensure_ascii=False) + "\n")
                            file_updated = True
                        else:
                            lines.append(line + "\n")
                            
                    except json.JSONDecodeError as e:
                        print(f"  跳过第{line_num}行，JSON解析错误: {e}")
                        lines.append(line + "\n")
                        error_count += 1
                    except Exception as e:
                        print(f"  处理第{line_num}行时出错: {e}")
                        lines.append(line + "\n")
                        error_count += 1
            
            # 如果文件有更新，重写文件
            if file_updated:
                with open(jsonl_file, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                print(f"  已更新文件: {jsonl_file}")
            
        except Exception as e:
            print(f"  处理文件 {jsonl_file} 时出错: {e}")
            error_count += 1
    
    # 同时修复元数据文件
    print("\n修复元数据文件...")
    images_dir = os.path.join(stats_logs_dir, "images")
    
    if os.path.exists(images_dir):
        for date_folder in os.listdir(images_dir):
            date_path = os.path.join(images_dir, date_folder)
            if not os.path.isdir(date_path):
                continue
                
            for record_folder in os.listdir(date_path):
                record_path = os.path.join(date_path, record_folder)
                if not os.path.isdir(record_path):
                    continue
                
                meta_file = os.path.join(record_path, "meta.json")
                if not os.path.exists(meta_file):
                    continue
                
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    
                    updated = False
                    ai_recheck = meta.get("ai_recheck", {})
                    
                    if ai_recheck.get("attempted", False):
                        ai_result = ai_recheck.get("ai_result", "")
                        
                        if ai_result in result_mapping:
                            english_result = result_mapping[ai_result]
                            meta["ai_recheck"]["ai_result"] = english_result
                            
                            # 如果复检成功，同时更新case_type
                            if ai_recheck.get("success", False) and english_result in ["fake_plate", "change_trailer", "normal"]:
                                old_case_type = meta.get("case_type", "")
                                meta["case_type"] = english_result
                                
                                print(f"  元数据 {record_folder}: "
                                      f"case_type {old_case_type} -> {english_result}, "
                                      f"AI结果 {ai_result} -> {english_result}")
                            
                            updated = True
                            fixed_count += 1
                    
                    if updated:
                        with open(meta_file, "w", encoding="utf-8") as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)
                
                except Exception as e:
                    print(f"  修复元数据 {record_folder} 时出错: {e}")
                    error_count += 1
    
    print(f"\n修复完成!")
    print(f"修复记录数: {fixed_count}")
    print(f"错误数: {error_count}")

if __name__ == "__main__":
    print("开始修复历史AI复检记录中的中文编码问题...")
    fix_history_ai_recheck()
