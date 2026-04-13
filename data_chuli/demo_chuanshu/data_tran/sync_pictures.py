import os
import json
import time
from datetime import datetime

import pandas as pd

from client import request_image


from data_output import connect_to_oracle


# 脚本所在目录，保证无论从哪里运行，进度文件路径都是固定的
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRESS_FILE ='D:\project\data_chuli\demo_chuanshu\data_tran\sync_progress.json'

# 本地根目录使用绝对盘符根路径
LOCAL_ROOT = 'D:\\'


def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return None, None
    try:
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('last_created_time'), data.get('last_task_id')
    except Exception:
        return None, None


def save_progress(created_time_str, task_id):
    data = {
        'last_created_time': created_time_str,
        'last_task_id': task_id,
    }
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def run_once():
    """执行一次从数据库增量拉取并下载图片的流程"""
    last_created_time, last_task_id = load_progress()
    print(f"当前进度: last_created_time={last_created_time}, last_task_id={last_task_id}")

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


def main(poll_interval_seconds: int = 300):
    """每隔 poll_interval_seconds 秒轮询一次数据库，自动下载新图片"""
    print(f"启动图片同步轮询服务，每 {poll_interval_seconds} 秒检查一次新数据...")
    try:
        while True:
            print("\n" + "#" * 60)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "开始一次轮询...")
            run_once()
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "本次轮询结束，进入休眠...")
            time.sleep(poll_interval_seconds)
    except KeyboardInterrupt:
        print("\n收到中断信号，停止轮询。")


if __name__ == '__main__':
    main()
