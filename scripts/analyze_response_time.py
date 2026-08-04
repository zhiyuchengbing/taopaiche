#!/usr/bin/env python3
"""
统计指定日期范围内请求耗时分布
"""
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

def analyze_response_times(log_dir, start_date_str, end_date_str):
    """
    统计请求耗时分布
    
    Args:
        log_dir: 日志目录路径
        start_date_str: 开始日期，格式 "YYYYMMDD"
        end_date_str: 结束日期，格式 "YYYYMMDD"
    """
    # 定义耗时区间（单位：毫秒）
    intervals = {
        "less_than_3s": {"count": 0, "range": "< 3s"},
        "3s_to_60s": {"count": 0, "range": "3s ~ 60s"},
        "60s_to_120s": {"count": 0, "range": "60s ~ 120s"},
        "greater_than_120s": {"count": 0, "range": "> 120s"}
    }
    
    # 按日期统计
    daily_stats = defaultdict(lambda: {
        "total": 0,
        "less_than_3s": 0,
        "3s_to_60s": 0,
        "60s_to_120s": 0,
        "greater_than_120s": 0
    })
    
    # 按case_type统计
    case_type_stats = defaultdict(lambda: {
        "total": 0,
        "less_than_3s": 0,
        "3s_to_60s": 0,
        "60s_to_120s": 0,
        "greater_than_120s": 0
    })
    
    total_requests = 0
    total_latency = 0
    
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")
    
    print(f"统计时间范围: {start_date_str} ~ {end_date_str}")
    print(f"日志目录: {log_dir}")
    print("-" * 60)
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        log_file_name = f"stats_{date_str}.jsonl"
        log_file_path = os.path.join(log_dir, log_file_name)
        
        if os.path.exists(log_file_path):
            print(f"处理文件: {log_file_name}")
            
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if not line.strip():
                            continue
                            
                        record = json.loads(line.strip())
                        
                        # 提取总耗时
                        lat_ms = record.get('lat_ms')
                        if lat_ms is None:
                            continue
                        
                        lat_s = lat_ms / 1000.0  # 转换为秒
                        total_requests += 1
                        total_latency += lat_ms
                        
                        # 提取日期和case_type
                        record_date = date_str
                        case_type = record.get('case_type', 'unknown')
                        
                        # 分类统计
                        if lat_s < 3:
                            interval_key = "less_than_3s"
                        elif 3 <= lat_s < 60:
                            interval_key = "3s_to_60s"
                        elif 60 <= lat_s < 120:
                            interval_key = "60s_to_120s"
                        else:
                            interval_key = "greater_than_120s"
                        
                        intervals[interval_key]["count"] += 1
                        daily_stats[record_date][interval_key] += 1
                        daily_stats[record_date]["total"] += 1
                        case_type_stats[case_type][interval_key] += 1
                        case_type_stats[case_type]["total"] += 1
                        
                    except json.JSONDecodeError as e:
                        print(f"  警告: 第{line_num}行JSON解析失败 - {e}")
                    except Exception as e:
                        print(f"  警告: 第{line_num}行处理失败 - {e}")
        else:
            print(f"文件不存在: {log_file_name}")
        
        current_date += timedelta(days=1)
    
    # 输出统计结果
    print("\n" + "=" * 60)
    print("总体统计")
    print("=" * 60)
    print(f"总请求数: {total_requests}")
    if total_requests > 0:
        avg_latency = total_latency / total_requests / 1000.0
        print(f"平均耗时: {avg_latency:.2f}秒")
    
    print("\n耗时区间分布:")
    for key, info in intervals.items():
        count = info["count"]
        percentage = (count / total_requests * 100) if total_requests > 0 else 0
        print(f"  {info['range']:15s}: {count:6d} 次 ({percentage:5.2f}%)")
    
    # 按日期统计
    print("\n" + "=" * 60)
    print("按日期统计")
    print("=" * 60)
    print(f"{'日期':<12} {'总数':>6} {'<3s':>8} {'3-60s':>8} {'60-120s':>8} {'>120s':>8}")
    print("-" * 60)
    
    for date in sorted(daily_stats.keys()):
        stats = daily_stats[date]
        print(f"{date:<12} {stats['total']:>6} "
              f"{stats['less_than_3s']:>8} {stats['3s_to_60s']:>8} "
              f"{stats['60s_to_120s']:>8} {stats['greater_than_120s']:>8}")
    
    # 按case_type统计
    print("\n" + "=" * 60)
    print("按判定类型统计")
    print("=" * 60)
    print(f"{'类型':<15} {'总数':>6} {'<3s':>8} {'3-60s':>8} {'60-120s':>8} {'>120s':>8}")
    print("-" * 60)
    
    for case_type in sorted(case_type_stats.keys()):
        stats = case_type_stats[case_type]
        print(f"{case_type:<15} {stats['total']:>6} "
              f"{stats['less_than_3s']:>8} {stats['3s_to_60s']:>8} "
              f"{stats['60s_to_120s']:>8} {stats['greater_than_120s']:>8}")
    
    # 导出CSV
    export_csv(daily_stats, case_type_stats, intervals, start_date_str, end_date_str)
    
    return intervals, daily_stats, case_type_stats

def export_csv(daily_stats, case_type_stats, intervals, start_date_str, end_date_str):
    """导出统计结果到CSV"""
    import csv
    
    # 导出按日期统计
    csv_file = f"response_time_stats_{start_date_str}_{end_date_str}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['日期', '总数', '<3s', '3-60s', '60-120s', '>120s'])
        for date in sorted(daily_stats.keys()):
            stats = daily_stats[date]
            writer.writerow([
                date, stats['total'], stats['less_than_3s'],
                stats['3s_to_60s'], stats['60s_to_120s'], stats['greater_than_120s']
            ])
    
    print(f"\n统计结果已导出到: {csv_file}")

if __name__ == "__main__":
    # 配置参数
    log_dir = r"data_chuli\demo\demo\Siamese-pytorch-master\stats_logs"
    start_date = "20260601"
    end_date = "20260610"
    
    analyze_response_times(log_dir, start_date, end_date)
