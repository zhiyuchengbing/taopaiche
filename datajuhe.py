import shutil
from pathlib import Path

def gather_files(src_dir: str, dst_dir: str) -> None:
    src = Path(src_dir).expanduser().resolve()
    dst = Path(dst_dir).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    counter = {}
    for path in src.rglob("*"):
        if path.is_file():
            name = path.name
            # 避免重名：同名则在后面加 _1, _2...
            idx = counter.get(name, 0)
            target_name = name if idx == 0 else f"{path.stem}_{idx}{path.suffix}"
            counter[name] = idx + 1
            shutil.copy2(path, dst / target_name)

if __name__ == "__main__":
    # 修改为你的源目录和目标目录
    gather_files(r"D:\data", r"D:\datajuhe")
    print("Done.")