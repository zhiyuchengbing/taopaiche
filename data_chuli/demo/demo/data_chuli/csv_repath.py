import argparse
import csv
import os
import sys
from typing import Optional
import re

DEFAULT_INPUT = r"E:\\套牌车识别项目\\demo\\20251122双视角数据\\匹配数据.csv"
DEFAULT_OUTPUT = r"E:\\套牌车识别项目\\demo\\20251122双视角数据\\匹配数据.csv"
DEFAULT_COLUMNS = r""
DEFAULT_OLD_ROOT = r"D:\\AlarmCaptures\\"
DEFAULT_NEW_ROOT = r"E:\\套牌车识别项目\\demo\\20251122双视角数据\\AlarmCaptures\\"
DEFAULT_ENCODING = "auto"
DEFAULT_DIALECT = "auto"
DEFAULT_CANONICAL_HEADERS = [
    "任务ID",
    "磅单号",
    "车号",
    "皮重过磅时间",
    "毛重过磅时间",
    "过皮部位1图片URL",
    "过皮部位2图片URL",
    "过毛部位1图片URL",
    "过毛部位2图片URL",
    "任务状态",
    "任务创建时间",
    "最后更新时间",
    "皮重磅房号",
    "毛重磅房号",
    "皮重重量",
    "毛重重量",
]

def parse_args():
    p = argparse.ArgumentParser(
        prog="csv_repath",
        description=(
            "Batch replace Windows absolute paths in specified CSV columns by"
            " swapping an old root with a new root, preserving the remainder."
        ),
    )
    p.add_argument("--input", "-i", required=False, default=None, help="Input CSV file path")
    p.add_argument(
        "--output",
        "-o",
        help="Output CSV file path (default: <input>.repath.csv in same folder)",
    )
    p.add_argument(
        "--extract-first-row-images",
        action="store_true",
        help="Extract and print all image-like paths from the first data row; do not modify CSV.",
    )
    p.add_argument(
        "--old-root",
        default=None,
        help="Old root prefix to replace (default: D:\\AlarmCaptures\\)",
    )
    p.add_argument(
        "--new-root",
        default=None,
        help=(
            "New root prefix (default: E:\\套牌车识别项目\\demo\\20251122双视角数据\\AlarmCaptures\\)"
        ),
    )
    p.add_argument(
        "--columns",
        "-c",
        help=(
            "Comma-separated list of column headers to process. If omitted,"
            " columns containing '图片URL' will be processed by default."
        ),
    )
    p.add_argument(
        "--encoding",
        default=None,
        help="CSV encoding for input and output (default: utf-8-sig)",
    )
    p.add_argument(
        "--dialect",
        choices=["auto", "excel", "excel-tab"],
        default=None,
        help="CSV dialect (default: auto detect)",
    )
    return p.parse_args()


def detect_dialect(path, encoding):
    with open(path, "r", encoding=encoding, newline="") as f:
        sample = f.read(32768)
    try:
        return csv.Sniffer().sniff(sample)
    except Exception:
        return csv.excel


def normalize_root(s: str) -> str:
    if not s:
        return s
    # Ensure Windows backslashes and trailing backslash
    s = s.replace("/", "\\")
    if not s.endswith("\\"):
        s += "\\"
    return s

def clean_header(name: str) -> str:
    if name is None:
        return ""
    # remove newlines, surrounding spaces, BOM/zero-width, quotes, and trailing Chinese colon
    s = str(name).replace("\r", "").replace("\n", "").strip()
    s = s.lstrip("\ufeff\u200b\u200c\u200d").strip('"')
    # unify full-width colon variant
    s = s.rstrip("：:").strip()
    return s


def resolve_encoding(path: str, encoding_opt: Optional[str]) -> str:
    """Pick a workable encoding for the CSV file.
    If encoding_opt is 'auto' or falsy, try common encodings sequentially.
    """
    candidates = []
    if encoding_opt and encoding_opt.lower() != "auto":
        candidates = [encoding_opt]
    else:
        candidates = [
            "utf-8-sig",
            "utf-8",
            "gbk",
            "gb2312",
            "big5",
            "mbcs",  # Windows ANSI codepage
        ]
    for enc in candidates:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                # read small chunk to validate
                f.read(1024)
            return enc
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            raise
        except Exception:
            # Other IO errors shouldn't trigger trying next enc
            continue
    # If all failed, fall back to utf-8-sig which will raise a clearer error later
    return "utf-8-sig"


def main():
    args = parse_args()

    in_path = args.input or DEFAULT_INPUT
    out_path = args.output if args.output else (DEFAULT_OUTPUT if DEFAULT_OUTPUT else None)
    old_root = normalize_root(args.old_root or DEFAULT_OLD_ROOT)
    new_root = normalize_root(args.new_root or DEFAULT_NEW_ROOT)
    encoding_opt = args.encoding or DEFAULT_ENCODING
    dialect_opt = (args.dialect or DEFAULT_DIALECT)

    if not os.path.isfile(in_path):
        print(f"Input CSV not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve encoding before any reads
    encoding = resolve_encoding(in_path, encoding_opt)

    if out_path is None:
        base, ext = os.path.splitext(os.path.basename(in_path))
        out_path = os.path.join(os.path.dirname(in_path), base + ".repath" + (ext or ".csv"))

    dialect = detect_dialect(in_path, encoding) if dialect_opt == "auto" else getattr(csv, dialect_opt.replace("-", "_"))

    # If only extracting image paths from the first data row, do so and exit
    if getattr(args, "extract_first_row_images", False):
        img_re = re.compile(r"^[A-Za-z]:\\\\.*\.(jpg|jpeg|png|bmp|gif|webp)$", re.IGNORECASE)
        images = []
        with open(in_path, "r", encoding=encoding, newline="") as rf:
            reader = csv.reader(rf, dialect=dialect)
            try:
                header = next(reader)
            except StopIteration:
                print("CSV is empty.")
                sys.exit(0)
            try:
                first = next(reader)
            except StopIteration:
                print("CSV has header only, no data rows.")
                sys.exit(0)
            for val in first:
                if not val:
                    continue
                s = str(val).replace("/", "\\").strip().lstrip("\ufeff\u200b\u200c\u200d")
                # Accept values that look like Windows absolute image paths
                if img_re.match(s):
                    images.append(s)
        if not images:
            print("No image-like paths found in the first data row.")
        else:
            print("Images in first data row:")
            for pth in images:
                exists = os.path.isfile(pth)
                print(f"  {pth}    [{'OK' if exists else 'MISSING'}]")
        return

    # Support multiple possible old roots (e.g., with/without trailing 's')
    candidate_old_roots = {old_root, normalize_root(r"D:\\AlarmCapture\\")}

    rows_out = []
    replaced_cells = 0
    examined_cells = 0
    demo_samples = []  # store up to 10 demo replacements
    raw_samples = []   # store up to 5 raw samples when no replacement

    with open(in_path, "r", encoding=encoding, newline="") as rf:
        reader = csv.reader(rf, dialect=dialect)
        for row in reader:
            new_row = []
            for val in row:
                if val is None or val == "":
                    new_row.append(val)
                    continue
                s = str(val)
                s_norm = s.replace("/", "\\").strip().lstrip("\ufeff\u200b\u200c\u200d")
                original = s_norm
                replaced = False
                for r in candidate_old_roots:
                    if s_norm.startswith(r):
                        suffix = s_norm[len(r):]
                        s_norm = new_root + suffix
                        replaced = True
                        break
                if not replaced:
                    for r in candidate_old_roots:
                        if r in s_norm:
                            s_norm = s_norm.replace(r, new_root, 1)
                            replaced = True
                            break
                new_row.append(s_norm)
                examined_cells += 1
                if replaced and s_norm != original:
                    replaced_cells += 1
                    if len(demo_samples) < 10:
                        demo_samples.append((original, s_norm))
                else:
                    if len(raw_samples) < 5:
                        raw_samples.append(repr(val))
            rows_out.append(new_row)

    with open(out_path, "w", encoding=encoding, newline="") as wf:
        writer = csv.writer(wf, dialect=dialect)
        writer.writerows(rows_out)

    print(f"Wrote: {out_path}")
    print(f"Processed cells: {examined_cells}, Replaced: {replaced_cells}")
    if demo_samples:
        print("Sample replacements (up to 10):")
        for before, after in demo_samples:
            print(f"  {before} -> {after}")
    elif raw_samples:
        print("No replacements were made. Here are sample raw values (repr) from target columns:")
        for raw in raw_samples:
            print(f"  {raw}")


if __name__ == "__main__":
    main()
