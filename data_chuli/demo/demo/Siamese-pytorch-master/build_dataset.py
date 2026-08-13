import os
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# 配置
EXPORTS_DIR = r"D:\project\data_chuli\demo\demo\Siamese-pytorch-master\exports"
DATASET_BASE_DIR = r"D:\test_dataset"
CATEGORIES = ["normal", "fake_plate", "change_trailer"]
EVAL_DISTRIBUTION = {"normal": 0.9, "change_trailer": 0.07, "fake_plate": 0.03}
EVAL_TOTAL = 500


def scan_exports_directory(exports_dir: str) -> List[Dict[str, Any]]:
    """扫描导出目录，提取已复核的记录"""
    records = []
    exports_path = Path(exports_dir)
    
    if not exports_path.exists():
        print(f"导出目录不存在: {exports_dir}")
        return records
    
    for record_dir in exports_path.iterdir():
        if not record_dir.is_dir():
            continue
            
        meta_path = record_dir / "meta.json"
        if not meta_path.exists():
            continue
            
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            # 只处理已复核的记录
            if not meta.get("reviewed", False):
                continue
            
            reviewed_case_type = meta.get("reviewed_case_type")
            if not reviewed_case_type or reviewed_case_type not in CATEGORIES:
                continue
            
            # 提取图片路径
            image_paths = {
                "path1": meta.get("input_path1"),
                "path2": meta.get("input_path2"),
                "path3": meta.get("input_path3"),
                "path4": meta.get("input_path4"),
            }
            
            # 验证主图片路径存在
            if not image_paths["path1"] or not image_paths["path2"]:
                continue
            
            record = {
                "record_id": meta.get("record_id"),
                "original_record_id": record_dir.name,
                "meta_path": str(meta_path),
                "image_paths": image_paths,
                "reviewed_case_type": reviewed_case_type,  # 人工复核结果
                "review_reason": meta.get("review_reason", ""),
                "reviewed_by": meta.get("reviewed_by"),
                "reviewed_at": meta.get("reviewed_at"),
            }
            records.append(record)
            
        except Exception as e:
            print(f"读取 {meta_path} 失败: {e}")
            continue
    
    print(f"扫描完成，找到 {len(records)} 条已复核记录")
    return records


def categorize_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按类别分类记录（以人工复核结果reviewed_case_type为准）"""
    categorized = {cat: [] for cat in CATEGORIES}
    
    for record in records:
        cat = record["reviewed_case_type"]
        categorized[cat].append(record)
    
    for cat in CATEGORIES:
        print(f"{cat}: {len(categorized[cat])} 条记录")
    
    return categorized


def copy_sample_to_category(record: Dict[str, Any], category_dir: Path, sample_id: str):
    """复制样本到类别目录"""
    sample_dir = category_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制图片
    image_paths = record["image_paths"]
    for key, src_path in image_paths.items():
        if not src_path:
            continue
        
        src_file = Path(src_path)
        if not src_file.exists():
            print(f"图片不存在: {src_path}")
            continue
        
        dst_file = sample_dir / f"{key}{src_file.suffix}"
        shutil.copy2(src_file, dst_file)
    
    # 生成简化的meta.json
    meta = {
        "sample_id": sample_id,
        "record_id": record.get("record_id"),
        "original_record_id": record["original_record_id"],
        "image_paths": {
            key: f"{key}{Path(src_path).suffix}" if src_path else None
            for key, src_path in image_paths.items()
        },
        "ground_truth": {
            "case_type": record["reviewed_case_type"],  # 人工复核结果（ground truth）
            "reason": record["review_reason"],
            "reviewed_by": record["reviewed_by"],
            "reviewed_at": record["reviewed_at"],
        }
    }
    
    meta_path = sample_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_category_datasets(categorized: Dict[str, List[Dict[str, Any]]], base_dir: str):
    """构建分类数据集"""
    base_path = Path(base_dir)
    
    for category in CATEGORIES:
        category_dir = base_path / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取现有最大 sample_id 和已存在的 original_record_id
        sample_counter = 1
        existing_original_ids = set()
        if category_dir.exists():
            for sample_dir in category_dir.iterdir():
                if sample_dir.is_dir() and sample_dir.name.startswith("sample_"):
                    try:
                        existing_id = int(sample_dir.name.split("_")[1])
                        if existing_id >= sample_counter:
                            sample_counter = existing_id + 1
                    except (ValueError, IndexError):
                        continue
                    
                    # 读取 meta.json 获取 original_record_id
                    meta_path = sample_dir / "meta.json"
                    if meta_path.exists():
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            original_id = meta.get("original_record_id")
                            if original_id:
                                existing_original_ids.add(original_id)
                        except Exception:
                            continue
        
        records = categorized[category]
        # 过滤掉已存在的记录
        new_records = [r for r in records if r["original_record_id"] not in existing_original_ids]
        
        for record in new_records:
            sample_id = f"sample_{sample_counter:04d}"
            sample_counter += 1
            copy_sample_to_category(record, category_dir, sample_id)
        
        print(f"{category} 数据集构建完成，新增 {len(new_records)} 条（跳过 {len(records) - len(new_records)} 条重复）")


def sample_for_evaluation(categorized: Dict[str, List[Dict[str, Any]]], 
                         base_dir: str, 
                         total: int = EVAL_TOTAL,
                         distribution: Dict[str, float] = EVAL_DISTRIBUTION) -> Dict[str, List[Dict[str, Any]]]:
    """抽样生成评估数据集"""
    sampled = {}
    
    for category in CATEGORIES:
        available = len(categorized[category])
        target_count = int(total * distribution.get(category, 0))
        
        if available == 0:
            print(f"{category} 没有可用数据")
            sampled[category] = []
            continue
        
        if available < target_count:
            print(f"{category} 数据不足，可用 {available}，目标 {target_count}，使用全部数据")
            sampled[category] = categorized[category].copy()
        else:
            sampled[category] = random.sample(categorized[category], target_count)
            print(f"{category} 抽取 {target_count} 条（可用 {available} 条）")
    
    return sampled


def build_evaluation_dataset(sampled: Dict[str, List[Dict[str, Any]]], base_dir: str, append: bool = False):
    """构建评估数据集"""
    base_path = Path(base_dir)
    eval_dir = base_path / "eval_dataset"
    samples_dir = eval_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果是追加模式，读取现有的dataset.json
    if append:
        dataset_json_path = eval_dir / "dataset.json"
        if dataset_json_path.exists():
            with open(dataset_json_path, "r", encoding="utf-8") as f:
                dataset_json = json.load(f)
            # 获取现有最大sample_id
            existing_samples = dataset_json.get("samples", [])
            if existing_samples:
                max_id = max(int(s["sample_id"].split("_")[1]) for s in existing_samples)
                sample_counter = max_id + 1
            else:
                sample_counter = 1
            print(f"追加模式：现有 {len(existing_samples)} 条样本，从 {sample_counter} 开始")
        else:
            dataset_json = {
                "dataset_id": f"eval_{datetime.now().strftime('%Y%m%d')}",
                "created_at": datetime.now().isoformat(),
                "total_samples": 0,
                "distribution": {},
                "samples": []
            }
            sample_counter = 1
    else:
        dataset_json = {
            "dataset_id": f"eval_{datetime.now().strftime('%Y%m%d')}",
            "created_at": datetime.now().isoformat(),
            "total_samples": 0,
            "distribution": {},
            "samples": []
        }
        sample_counter = 1
    
    for category in CATEGORIES:
        records = sampled[category]
        dataset_json["distribution"][category] = len(records)
        
        for record in records:
            sample_id = f"sample_{sample_counter:04d}"
            sample_counter += 1
            
            sample_dir = samples_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制图片
            image_paths = record["image_paths"]
            for key, src_path in image_paths.items():
                if not src_path:
                    continue
                
                src_file = Path(src_path)
                if not src_file.exists():
                    print(f"图片不存在: {src_path}")
                    continue
                
                dst_file = sample_dir / f"{key}{src_file.suffix}"
                shutil.copy2(src_file, dst_file)
            
            # 生成meta.json
            meta = {
                "sample_id": sample_id,
                "record_id": record.get("record_id"),
                "original_record_id": record["original_record_id"],
                "image_paths": {
                    key: f"{key}{Path(src_path).suffix}" if src_path else None
                    for key, src_path in image_paths.items()
                },
                "ground_truth": {
                    "case_type": record["reviewed_case_type"],  # 人工复核结果（ground truth）
                    "reason": record["review_reason"],
                    "reviewed_by": record["reviewed_by"],
                    "reviewed_at": record["reviewed_at"],
                }
            }
            
            meta_path = sample_dir / "meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            
            dataset_json["samples"].append({
                "sample_id": sample_id,
                "case_type": category,
                "meta_path": f"samples/{sample_id}/meta.json"
            })
    
    dataset_json["total_samples"] = len(dataset_json["samples"])
    
    # 保存dataset.json
    dataset_json_path = eval_dir / "dataset.json"
    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, ensure_ascii=False, indent=2)
    
    print(f"评估数据集构建完成，共 {dataset_json['total_samples']} 条")
    print(f"分布: {dataset_json['distribution']}")


def build_single_category_dataset(category: str):
    """构建单个类别数据集"""
    print(f"开始构建 {category} 数据集...")
    
    # 扫描导出目录
    records = scan_exports_directory(EXPORTS_DIR)
    if not records:
        print("没有找到已复核记录")
        return
    
    # 分类
    categorized = categorize_records(records)
    
    # 只构建指定类别
    category_records = categorized.get(category, [])
    if not category_records:
        print(f"{category} 没有可用数据")
        return
    
    # 构建该类别数据集
    base_path = Path(DATASET_BASE_DIR)
    category_dir = base_path / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取现有最大 sample_id 和已存在的 original_record_id
    sample_counter = 1
    existing_original_ids = set()
    if category_dir.exists():
        for sample_dir in category_dir.iterdir():
            if sample_dir.is_dir() and sample_dir.name.startswith("sample_"):
                try:
                    existing_id = int(sample_dir.name.split("_")[1])
                    if existing_id >= sample_counter:
                        sample_counter = existing_id + 1
                except (ValueError, IndexError):
                    continue
                
                # 读取 meta.json 获取 original_record_id
                meta_path = sample_dir / "meta.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        original_id = meta.get("original_record_id")
                        if original_id:
                            existing_original_ids.add(original_id)
                    except Exception:
                        continue
    
    # 过滤掉已存在的记录
    new_records = [r for r in category_records if r["original_record_id"] not in existing_original_ids]
    
    for record in new_records:
        sample_id = f"sample_{sample_counter:04d}"
        sample_counter += 1
        copy_sample_to_category(record, category_dir, sample_id)
    
    print(f"{category} 数据集构建完成，新增 {len(new_records)} 条（跳过 {len(category_records) - len(new_records)} 条重复）")


def build_all_category_datasets() -> Dict[str, int]:
    """构建所有类别数据集并返回统计信息"""
    print("开始构建所有分类数据集...")
    
    # 扫描导出目录
    records = scan_exports_directory(EXPORTS_DIR)
    if not records:
        print("没有找到已复核记录")
        return {"total": 0, "normal": 0, "fake_plate": 0, "change_trailer": 0}
    
    # 分类
    categorized = categorize_records(records)
    
    # 构建所有类别数据集
    build_category_datasets(categorized, DATASET_BASE_DIR)
    
    # 返回统计信息
    stats = {
        "total": len(records),
        "normal": len(categorized.get("normal", [])),
        "fake_plate": len(categorized.get("fake_plate", [])),
        "change_trailer": len(categorized.get("change_trailer", []))
    }
    
    print(f"所有分类数据集构建完成，统计: {stats}")
    return stats


def build_eval_dataset_only():
    """仅构建评估数据集（从已有的分类数据集中抽样）"""
    print("开始构建评估数据集...")
    
    # 读取已存在的评估数据集record_id，避免重复
    base_path = Path(DATASET_BASE_DIR)
    eval_dir = base_path / "eval_dataset"
    existing_record_ids = set()
    
    if eval_dir.exists():
        samples_dir = eval_dir / "samples"
        if samples_dir.exists():
            for sample_dir in samples_dir.iterdir():
                if not sample_dir.is_dir() or not sample_dir.name.startswith("sample_"):
                    continue
                
                meta_path = sample_dir / "meta.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        record_id = meta.get("record_id")
                        if record_id:
                            existing_record_ids.add(record_id)
                    except Exception:
                        continue
    
    print(f"已存在 {len(existing_record_ids)} 条评估数据记录")
    
    # 扫描已有的分类数据集
    categorized = {}
    total_available = 0
    
    for category in CATEGORIES:
        category_dir = base_path / category
        if not category_dir.exists():
            print(f"{category} 数据集不存在，请先构造")
            categorized[category] = []
            continue
        
        # 从分类数据集读取
        records = []
        for sample_dir in category_dir.iterdir():
            if not sample_dir.is_dir() or not sample_dir.name.startswith("sample_"):
                continue
            
            meta_path = sample_dir / "meta.json"
            if not meta_path.exists():
                continue
            
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                record_id = meta.get("record_id")
                # 跳过已存在的记录
                if record_id and record_id in existing_record_ids:
                    continue
                
                # 构造完整的图片路径（从分类数据集样本目录）
                full_image_paths = {}
                for key, rel_path in meta["image_paths"].items():
                    if rel_path:
                        full_image_paths[key] = str(sample_dir / rel_path)
                
                record = {
                    "record_id": record_id,
                    "original_record_id": meta.get("original_record_id"),
                    "meta_path": str(meta_path),
                    "image_paths": full_image_paths,
                    "reviewed_case_type": meta["ground_truth"]["case_type"],
                    "review_reason": meta["ground_truth"].get("reason", ""),
                    "reviewed_by": meta["ground_truth"].get("reviewed_by"),
                    "reviewed_at": meta["ground_truth"].get("reviewed_at"),
                }
                records.append(record)
            except Exception as e:
                print(f"读取 {meta_path} 失败: {e}")
                continue
        
        categorized[category] = records
        total_available += len(records)
        print(f"{category}: 读取 {len(records)} 条记录（排除已存在的）")
    
    # 检查是否有可用数据
    if total_available == 0:
        raise ValueError("没有可用的分类数据集，请先构造分类数据集")
    
    # 抽样
    sampled = sample_for_evaluation(categorized, DATASET_BASE_DIR, EVAL_TOTAL, EVAL_DISTRIBUTION)
    
    # 构建评估数据集（追加模式）
    build_evaluation_dataset(sampled, DATASET_BASE_DIR, append=True)
    
    print("评估数据集构建完成！")


def main():
    print("开始构建评估数据集...")
    
    # 扫描导出目录
    records = scan_exports_directory(EXPORTS_DIR)
    if not records:
        print("没有找到已复核记录")
        return
    
    # 分类
    categorized = categorize_records(records)
    
    # 构建分类数据集
    build_category_datasets(categorized, DATASET_BASE_DIR)
    
    # 抽样
    sampled = sample_for_evaluation(categorized, DATASET_BASE_DIR)
    
    # 构建评估数据集
    build_evaluation_dataset(sampled, DATASET_BASE_DIR)
    
    print("数据集构建完成！")


if __name__ == "__main__":
    main()
