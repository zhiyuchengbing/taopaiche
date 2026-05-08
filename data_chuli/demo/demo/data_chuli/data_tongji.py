# analyze_plates.py
import pandas as pd
from pathlib import Path

# 配置你的 CSV 路径
csv_path = Path(r"e:\套牌车识别项目\demo\demo\data_chuli\data\匹配数据.csv")
out_dir = csv_path.parent  # 输出到同目录

# 尝试多种编码，适配 Windows/中文 CSV
encodings = ["utf-8-sig", "gbk", "utf-8"]
last_err = None
for enc in encodings:
    try:
        df = pd.read_csv(csv_path, encoding=enc)
        break
    except Exception as e:
        last_err = e
else:
    raise RuntimeError(f"无法读取文件，请检查编码/路径。最后错误: {last_err}")

# 必要列名
plate_col = "车号"
if plate_col not in df.columns:
    raise KeyError(f"未找到必须列: {plate_col}。实际列名: {list(df.columns)}")

# 计数
counts = (
    df[plate_col]
    .fillna("UNKNOWN")
    .astype(str)
    .str.strip()
    .value_counts(dropna=False)
    .rename_axis(plate_col)
    .reset_index(name="count")
)

# 标记重复（count > 1）
dupe_plates = set(counts.loc[counts["count"] > 1, plate_col])

# 重复明细（按车号筛选）
duplicates_rows = df[df[plate_col].astype(str).str.strip().isin(dupe_plates)].copy()

# 保存结果
counts_path = out_dir / "plate_counts.csv"
dupes_path = out_dir / "duplicate_rows.csv"
counts.to_csv(counts_path, index=False, encoding="utf-8-sig")
duplicates_rows.to_csv(dupes_path, index=False, encoding="utf-8-sig")

# 控制台摘要
total_unique = counts.shape[0]
duplicates_num = (counts["count"] > 1).sum()
print(f"总唯一车牌数: {total_unique}")
print(f"存在重复的车牌数: {duplicates_num}")
if duplicates_num:
    print("重复车牌及次数（Top 20）：")
    print(counts[counts['count'] > 1].head(20).to_string(index=False))
print(f"已保存: {counts_path}")
print(f"已保存: {dupes_path}")