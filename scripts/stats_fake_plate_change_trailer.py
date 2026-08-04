#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线统计套牌和换挂记录脚本
统计指定时间段内判定为套牌(fake_plate)或换挂(change_trailer)的记录
只统计记录ID和meta.json中的final_diff_summary字段
分别导出到不同的Excel文件
"""

import os
import json
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any


def get_date_range_log_files(log_dir: str, start_date_str: str, end_date_str: str) -> List[str]:
    """获取指定日期范围的日志文件"""
    log_files = []
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"stats_{date_str}.jsonl")
        if os.path.exists(log_file):
            log_files.append(log_file)
        current_date += timedelta(days=1)
    
    return sorted(log_files)


def parse_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """解析jsonl文件"""
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    return records


def filter_records_by_type(records: List[Dict[str, Any]], case_type: str) -> List[Dict[str, Any]]:
    """筛选指定类型的记录"""
    return [r for r in records if r.get('case_type') == case_type]


def read_meta_json(image_dir: str, record_id: str) -> Dict[str, Any]:
    """读取meta.json文件"""
    try:
        meta_path = os.path.join(image_dir, record_id, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"读取meta.json出错 {record_id}: {e}")
    return {}


def extract_record_info(record: Dict[str, Any], images_base_dir: str) -> Dict[str, Any]:
    """从记录中提取需要的信息"""
    record_id = record.get('record_id', '')
    case_type = record.get('case_type', '')
    ts = record.get('ts', '')
    image_dir = record.get('image_dir', '')
    
    # 尝试从meta.json读取final_diff_summary
    final_diff_summary = ""
    if image_dir:
        meta_data = read_meta_json(image_dir, record_id)
        final_diff_summary = meta_data.get('final_diff_summary', '')
    
    # 如果meta.json中没有final_diff_summary，尝试从jsonl的diff_desc字段获取
    if not final_diff_summary:
        final_diff_summary = record.get('diff_desc', '')
    
    return {
        'record_id': record_id,
        'case_type': case_type,
        'ts': ts,
        'final_diff_summary': final_diff_summary
    }


def export_to_excel(records: List[Dict[str, Any]], case_type: str, output_file: str):
    """导出记录到Excel文件(CSV格式)"""
    type_name = '套牌' if case_type == 'fake_plate' else '换挂'
    
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['记录ID', '时间', '差异说明(final_diff_summary)'])
        
        for record in records:
            writer.writerow([
                record['record_id'],
                record['ts'],
                record['final_diff_summary']
            ])
    
    print(f"{type_name}记录已导出到: {output_file}")
    return len(records)


def generate_daily_statistics_table(records: List[Dict[str, Any]], start_date_str: str, end_date_str: str):
    """生成按日期统计的表格"""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # 按日期统计
    daily_stats = {}
    for record in records:
        ts = record.get('ts', '')
        if ts:
            try:
                date_str = ts[:10]  # YYYY-MM-DD
                if date_str not in daily_stats:
                    daily_stats[date_str] = {
                        'total': 0,
                        'normal': 0,
                        'fake_plate': 0,
                        'change_trailer': 0
                    }
                daily_stats[date_str]['total'] += 1
                case_type = record.get('case_type', '')
                if case_type in daily_stats[date_str]:
                    daily_stats[date_str][case_type] += 1
                else:
                    daily_stats[date_str]['normal'] += 1
            except:
                pass
    
    # 生成表格
    print("\n" + "="*80)
    print("每日统计表格 (6月1日 - 6月10日)")
    print("="*80)
    print(f"{'日期':<12} {'请求总数':<10} {'正常':<10} {'套牌':<10} {'换挂':<10}")
    print("-"*80)
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        display_date = current_date.strftime("%m-%d")
        
        if date_str in daily_stats:
            stats = daily_stats[date_str]
            print(f"{display_date:<12} {stats['total']:<10} {stats['normal']:<10} {stats['fake_plate']:<10} {stats['change_trailer']:<10}")
        else:
            print(f"{display_date:<12} {'0':<10} {'0':<10} {'0':<10} {'0':<10}")
        
        current_date += timedelta(days=1)
    
    print("="*80)
    
    # 计算总计
    total_all = sum(s['total'] for s in daily_stats.values())
    total_normal = sum(s['normal'] for s in daily_stats.values())
    total_fake = sum(s['fake_plate'] for s in daily_stats.values())
    total_change = sum(s['change_trailer'] for s in daily_stats.values())
    
    print(f"{'总计':<12} {total_all:<10} {total_normal:<10} {total_fake:<10} {total_change:<10}")
    print("="*80)
    
    return daily_stats


def main():
    # 配置
    log_dir = r"d:\project\data_chuli\demo\demo\Siamese-pytorch-master\stats_logs"
    start_date = "2026-06-01"  # 开始日期
    end_date = "2026-06-10"    # 结束日期
    output_dir = r"d:\project"
    
    fake_plate_output = os.path.join(output_dir, "套牌记录_20260601_20260610.csv")
    change_trailer_output = os.path.join(output_dir, "换挂记录_20260601_20260610.csv")
    
    print("="*60)
    print(f"统计时间范围: {start_date} 到 {end_date}")
    print("="*60)
    
    # 获取日志文件
    print(f"\n正在查找 {start_date} 到 {end_date} 的日志文件...")
    log_files = get_date_range_log_files(log_dir, start_date, end_date)
    print(f"找到 {len(log_files)} 个日志文件")
    
    if not log_files:
        print("未找到任何日志文件")
        return
    
    # 读取并解析所有记录
    print("\n正在读取日志文件...")
    all_records = []
    for log_file in log_files:
        print(f"  处理: {os.path.basename(log_file)}")
        records = parse_jsonl_file(log_file)
        all_records.extend(records)
    
    print(f"总共读取 {len(all_records)} 条记录")
    
    # 生成每日统计表格
    print("\n正在生成每日统计表格...")
    generate_daily_statistics_table(all_records, start_date, end_date)
    
    # 分别筛选套牌和换挂记录
    print("\n正在筛选套牌记录...")
    fake_plate_records = filter_records_by_type(all_records, 'fake_plate')
    print(f"筛选出 {len(fake_plate_records)} 条套牌记录")
    
    print("\n正在筛选换挂记录...")
    change_trailer_records = filter_records_by_type(all_records, 'change_trailer')
    print(f"筛选出 {len(change_trailer_records)} 条换挂记录")
    
    # 提取记录信息
    print("\n正在提取记录信息...")
    fake_plate_info = [extract_record_info(r, log_dir) for r in fake_plate_records]
    change_trailer_info = [extract_record_info(r, log_dir) for r in change_trailer_records]
    
    # 导出到不同的文件
    print("\n正在导出Excel文件...")
    fake_count = export_to_excel(fake_plate_info, 'fake_plate', fake_plate_output)
    change_count = export_to_excel(change_trailer_info, 'change_trailer', change_trailer_output)
    
    # 打印统计摘要
    print("\n" + "="*60)
    print("统计完成!")
    print("="*60)
    print(f"套牌记录: {fake_count} 条")
    print(f"换挂记录: {change_count} 条")
    print(f"总计: {fake_count + change_count} 条")
    print("="*60)


if __name__ == "__main__":
    main()
