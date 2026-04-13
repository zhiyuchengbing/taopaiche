"""
# 图片同步服务 - 功能小结

## 功能概述
本脚本用于从 Oracle 数据库增量拉取图片任务记录，并通过网络从服务器下载对应的图片文件到本地。

## 核心功能

### 1. 增量数据同步
- 从 `jlyxz.PIC_MATCHTASK` 表查询新的任务记录
- 支持基于 `CREATED_TIME` 和 `TASK_ID` 的增量查询
- 自动保存处理进度，支持断点续传

### 2. 图片下载
- 支持下载每个任务的 8 张图片（4 张 TARE 图片 + 4 张 GROSS 图片）
- 自动创建本地目录结构
- 跳过已存在的文件，避免重复下载
- 通过 TCP Socket 从服务器下载图片文件

### 3. 自动重启机制
- **宕机自动重启**：捕获所有未处理的异常，自动重启程序
- **长时间无下载自动重启**：超过 2 小时没有下载任何文件时自动重启
- 重启前等待 10 秒，便于查看日志

## 配置参数

- `MAX_NO_DOWNLOAD_HOURS = 2`：超过多少小时没有下载就重启（默认 2 小时）
- `RESTART_DELAY_SECONDS = 10`：重启前等待的秒数
- `LOCAL_ROOT = 'D:\\'`：本地保存根目录
- `PROGRESS_FILE`：进度文件路径，保存最后处理的时间和任务ID

## 工作流程

1. 启动服务，每 300 秒（5分钟）轮询一次数据库
2. 查询新的任务记录（基于上次处理进度）
3. 对每条记录，下载所有相关的图片文件
4. 每处理完一条记录，立即保存进度
5. 每次轮询前检查是否需要重启（长时间无下载）
6. 发生异常时自动重启程序

## 进度跟踪

进度文件 `sync_progress.json` 包含：
- `last_created_time`：最后处理的记录创建时间
- `last_task_id`：最后处理的任务ID
- `last_download_time`：最后一次成功下载的时间

## 使用说明

直接运行脚本即可：
```bash
python sync_pictures.py
```

程序会持续运行，自动处理新的图片任务。使用 `Ctrl+C` 可以正常退出。
"""

import os
import json
import time
import sys
import subprocess
from datetime import datetime, timedelta

import pandas as pd

from client import request_image


from data_output import connect_to_oracle


# 脚本所在目录，保证无论从哪里运行，进度文件路径都是固定的
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRESS_FILE ='D:\project\data_tran\sync_progress.json'

# 本地根目录使用绝对盘符根路径
LOCAL_ROOT = 'D:\\'

# 配置参数
MAX_NO_DOWNLOAD_HOURS = 2  # 超过多少小时没有下载就重启（默认2小时）
RESTART_DELAY_SECONDS = 10  # 重启前等待的秒数


def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return None, None, None
    try:
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('last_created_time'), data.get('last_task_id'), data.get('last_download_time')
    except Exception:
        return None, None, None


def save_progress(created_time_str, task_id, download_time_str=None):
    data = {
        'last_created_time': created_time_str,
        'last_task_id': task_id,
    }
    if download_time_str:
        data['last_download_time'] = download_time_str
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_download_time():
    """更新最后一次下载时间"""
    last_created_time, last_task_id, _ = load_progress()
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_progress(last_created_time, last_task_id, current_time_str)


def ensure_local_path(remote_path):
    drive, path_without_drive = os.path.splitdrive(remote_path)
    rel_path = path_without_drive.lstrip('\\/')
    # 如果 LOCAL_ROOT 是类似 "D:" 这样的盘符，拼成真正的根目录 "D:\\"
    if LOCAL_ROOT.endswith(':'):
        base = LOCAL_ROOT + os.sep
    else:
        base = LOCAL_ROOT

    local_path = os.path.join(base, rel_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    return local_path


def fetch_new_rows(connection, last_created_time, last_task_id):
    base_sql = """
    SELECT TASK_ID,
           CREATED_TIME,
           TARE_IMAGE_PATH1, TARE_IMAGE_PATH2, TARE_IMAGE_PATH3, TARE_IMAGE_PATH4,
           GROSS_IMAGE_PATH1, GROSS_IMAGE_PATH2, GROSS_IMAGE_PATH3, GROSS_IMAGE_PATH4
    FROM jlyxz.PIC_MATCHTASK
    """

    if last_created_time is None:
        where_clause = ""
        params = {}
    else:
        where_clause = """
        WHERE (CREATED_TIME > :last_time)
           OR (CREATED_TIME = :last_time AND TASK_ID > :last_task_id)
        """
        params = {
            'last_time': datetime.strptime(last_created_time, '%Y-%m-%d %H:%M:%S'),
            'last_task_id': last_task_id or '0',
        }

    order_clause = "ORDER BY CREATED_TIME, TASK_ID"
    sql = "\n".join([base_sql, where_clause, order_clause])

    cursor = connection.cursor()
    cursor.execute(sql, **params)
    columns = [c[0] for c in cursor.description]
    data = cursor.fetchall()
    cursor.close()

    if not data:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(data, columns=columns)


def download_for_row(row, server_ip='10.100.2.229', server_port=5000):
    paths = [
        row.get('TARE_IMAGE_PATH1'), row.get('TARE_IMAGE_PATH2'),
        row.get('TARE_IMAGE_PATH3'), row.get('TARE_IMAGE_PATH4'),
        row.get('GROSS_IMAGE_PATH1'), row.get('GROSS_IMAGE_PATH2'),
        row.get('GROSS_IMAGE_PATH3'), row.get('GROSS_IMAGE_PATH4'),
    ]

    has_successful_download = False

    for remote_path in paths:
        if not remote_path or not isinstance(remote_path, str):
            continue

        remote_path = remote_path.strip()
        if not remote_path:
            continue

        local_path = ensure_local_path(remote_path)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            print(f"已存在，跳过: {local_path}")
            continue

        save_dir = os.path.dirname(local_path)
        os.makedirs(save_dir, exist_ok=True)

        print(f"请求服务器文件: {remote_path}")
        print(f"本地保存到: {local_path}")

        success = request_image(server_ip, server_port, remote_path, save_dir)
        if not success:
            print(f"下载失败: {remote_path}")
        else:
            print(f"下载成功: {remote_path}")
            has_successful_download = True

    # 如果有成功下载，更新下载时间
    if has_successful_download:
        update_download_time()


def run_once():
    """执行一次从数据库增量拉取并下载图片的流程"""
    last_created_time, last_task_id, last_download_time = load_progress()
    print(f"当前进度: last_created_time={last_created_time}, last_task_id={last_task_id}")
    print(f"最后下载时间: {last_download_time}")

    connection = connect_to_oracle()
    if not connection:
        print("无法连接到数据库，本次轮询结束")
        return

    try:
        df = fetch_new_rows(connection, last_created_time, last_task_id)
        if df.empty:
            print("没有新的记录需要处理")
            return

        print(f"本次需处理 {len(df)} 条记录")

        latest_created_time = last_created_time
        latest_task_id = last_task_id

        for _, row in df.iterrows():
            created_time = row['CREATED_TIME']
            task_id = str(row['TASK_ID'])

            print("\n" + "=" * 60)
            print(f"处理 TASK_ID={task_id}, CREATED_TIME={created_time}")

            download_for_row(row)

            if isinstance(created_time, datetime):
                created_time_str = created_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                created_time_str = str(created_time)

            latest_created_time = created_time_str
            latest_task_id = task_id
            save_progress(latest_created_time, latest_task_id)

        print("\n本次处理完成")
        print(f"最新进度: last_created_time={latest_created_time}, last_task_id={latest_task_id}")

    finally:
        connection.close()


def check_need_restart():
    """检查是否需要重启（长时间没有下载）"""
    _, _, last_download_time = load_progress()
    if last_download_time is None:
        # 如果从未下载过，不重启
        return False
    
    try:
        last_download = datetime.strptime(last_download_time, '%Y-%m-%d %H:%M:%S')
        now = datetime.now()
        hours_since_download = (now - last_download).total_seconds() / 3600
        
        if hours_since_download >= MAX_NO_DOWNLOAD_HOURS:
            print(f"\n警告: 已经 {hours_since_download:.2f} 小时没有下载任何文件")
            print(f"超过阈值 {MAX_NO_DOWNLOAD_HOURS} 小时，将重启程序...")
            return True
    except Exception as e:
        print(f"检查下载时间时出错: {e}")
    
    return False


def main(poll_interval_seconds: int = 300):
    """每隔 poll_interval_seconds 秒轮询一次数据库，自动下载新图片"""
    print(f"启动图片同步轮询服务，每 {poll_interval_seconds} 秒检查一次新数据...")
    print(f"配置: 超过 {MAX_NO_DOWNLOAD_HOURS} 小时无下载将自动重启")
    
    try:
        while True:
            # 检查是否需要重启（长时间无下载）
            if check_need_restart():
                print(f"\n等待 {RESTART_DELAY_SECONDS} 秒后重启...")
                time.sleep(RESTART_DELAY_SECONDS)
                restart_program()
            
            print("\n" + "#" * 60)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "开始一次轮询...")
            run_once()
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "本次轮询结束，进入休眠...")
            time.sleep(poll_interval_seconds)
    except KeyboardInterrupt:
        print("\n收到中断信号，停止轮询。")
    except Exception as e:
        print(f"\n发生未捕获的异常: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n等待 {RESTART_DELAY_SECONDS} 秒后自动重启...")
        time.sleep(RESTART_DELAY_SECONDS)
        restart_program()


def restart_program():
    """重启程序，使用 subprocess 确保环境变量被正确清理"""
    print("\n" + "=" * 60)
    print("正在重启程序...")
    print("=" * 60 + "\n")
    
    # 构建清理后的环境变量字典
    new_env = os.environ.copy()
    
    # 清理 PATH，只保留必要的路径
    try:
        essential_paths = [
            r"C:\Windows\system32",
            r"C:\Windows",
            r"C:\Windows\System32\Wbem",
        ]
        python_dir = os.path.dirname(sys.executable)
        if python_dir:
            essential_paths.append(python_dir)
            scripts_dir = os.path.join(python_dir, "Scripts")
            if os.path.exists(scripts_dir):
                essential_paths.append(scripts_dir)
        
        # 过滤掉不存在的路径
        cleaned_paths = [p for p in essential_paths if os.path.exists(p)]
        new_env["PATH"] = ";".join(cleaned_paths)
        
        current_path_len = len(new_env.get("PATH", ""))
        print(f"清理后的 PATH 长度为 {current_path_len} 字符")
    except Exception as e:
        print(f"清理 PATH 时出错: {e}，使用最小 PATH...")
        # 如果清理失败，至少设置一个最小的 PATH
        python_dir = os.path.dirname(sys.executable)
        if python_dir:
            new_env["PATH"] = python_dir
    
    # 使用 subprocess 启动新进程，明确指定清理后的环境变量
    python = sys.executable
    script_path = os.path.abspath(__file__)
    
    try:
        # 启动新进程
        subprocess.Popen([python, script_path], env=new_env)
        # 退出当前进程
        sys.exit(0)
    except Exception as e:
        print(f"使用 subprocess 重启失败: {e}，尝试使用 os.execl...")
        # 如果 subprocess 失败，回退到 os.execl（但环境变量可能仍然有问题）
        try:
            # 先尝试清理当前进程的环境变量
            if "PATH" in os.environ:
                try:
                    del os.environ["PATH"]
                except Exception:
                    pass
            os.environ["PATH"] = new_env.get("PATH", "")
            os.execl(python, python, *sys.argv)
        except Exception as e2:
            print(f"os.execl 也失败: {e2}")
            print("无法重启程序，请手动重启")
            sys.exit(1)


if __name__ == '__main__':
    main()
