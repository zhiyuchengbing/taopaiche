import os
import sys
import json
import time
from datetime import datetime

import socket
import pandas as pd
import cx_Oracle

from siamese import Siamese

# 保持与 GUI 脚本一致的路径设置，确保可以导入 data_chuli.cropper
parent_dir = os.path.dirname(os.path.dirname(__file__))  # ...\data_chuli\demo
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data_chuli.cropper import VehicleCropper


# 进度文件，记录已处理到的 GROSS_WEIGH_TIME 和 TASK_ID
PROGRESS_FILE = r"D:\project\data_chuli\auto_predict_progress.json"

# 自动输出的疑似套牌结果 CSV（追加写入）
AUTO_CSV_FILE = r"D:\project\suspected_fake_plate_auto.csv"

# 图片服务器配置（与 sync_pictures/client 中保持一致）
SERVER_IP = "10.100.2.229"
SERVER_PORT = 5000

# 本地根目录使用绝对盘符根路径（与 sync_pictures 中逻辑一致）
LOCAL_ROOT = 'D:\\'


def connect_to_oracle():
    """创建 Oracle 数据库连接（复制自 d:\project\data_output.py）"""
    try:
        os.environ["PATH"] = r"D:\\instantclient-basic-windows.x64-23.26.0.0.0\\instantclient_23_0" + ";" + os.environ.get("PATH", "")
        os.environ["TNS_ADMIN"] = r"D:\\instantclient-basic-windows.x64-23.26.0.0.0\\instantclient_23_0\\network\\admin"

        dsn_tns = cx_Oracle.makedsn('10.100.2.229', '1521', service_name='JLYXZ')
        connection = cx_Oracle.connect(user='identify', password='123456', dsn=dsn_tns)
        print("成功连接到Oracle数据库")
        return connection
    except Exception as e:
        print(f"连接数据库时出错: {e}")
        return None


def ensure_local_path(remote_path):
    """根据远程路径生成本地保存路径，并确保目录存在（简化自 sync_pictures.ensure_local_path）"""
    drive, path_without_drive = os.path.splitdrive(remote_path)
    rel_path = path_without_drive.lstrip('\\/')

    if LOCAL_ROOT.endswith(':'):
        base = LOCAL_ROOT + os.sep
    else:
        base = LOCAL_ROOT

    local_path = os.path.join(base, rel_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    return local_path


def receive_image(s, save_dir='.'):
    """从 socket 接收图片并保存（简化自 client.receive_image）"""
    try:
        file_name_length = int.from_bytes(s.recv(4), byteorder='big')
        if file_name_length == 0:
            print("服务器返回: 文件不存在或无法读取")
            return False
        file_name = s.recv(file_name_length).decode('utf-8')

        file_size = int.from_bytes(s.recv(8), byteorder='big')
        if file_size == 0:
            print("服务器返回: 文件大小为 0，可能不存在或无法读取")
            return False

        received_data = bytearray()
        print(f"正在接收图片: {file_name} ({file_size} 字节)")

        while len(received_data) < file_size:
            chunk = s.recv(min(4096, file_size - len(received_data)))
            if not chunk:
                raise Exception("连接中断")
            received_data.extend(chunk)

        save_path = os.path.join(save_dir, file_name)
        with open(save_path, 'wb') as f:
            f.write(received_data)

        print(f"图片已保存到: {os.path.abspath(save_path)}")
        return True

    except Exception as e:
        print(f"接收图片时出错: {e}")
        return False


def request_image(server_ip, server_port, image_name, save_dir='.'):
    """向图片服务器请求并接收图片（简化自 client.request_image）"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            print(f"正在连接到服务器 {server_ip}:{server_port}...")
            s.connect((server_ip, server_port))
            print("已连接到服务器")

            s.sendall(len(image_name).to_bytes(4, byteorder='big'))
            s.sendall(image_name.encode('utf-8'))

            return receive_image(s, save_dir)

        except Exception as e:
            print(f"连接服务器时出错: {e}")
            return False


def load_progress():
    """读取上次处理进度"""
    if not os.path.exists(PROGRESS_FILE):
        return None, None
    try:
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('last_gross_time'), data.get('last_task_id')
    except Exception:
        return None, None


def save_progress(gross_time_str, task_id):
    """保存本次处理结束时的进度"""
    data = {
        'last_gross_time': gross_time_str,
        'last_task_id': task_id,
    }
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_new_rows(connection, last_gross_time, last_task_id):
    """按 GROSS_WEIGH_TIME、TASK_ID 增量获取需要新处理的记录"""
    base_sql = """
    SELECT TASK_ID,
           TRUCK_ID,
           GROSS_WEIGH_TIME,
           TARE_IMAGE_PATH1,
           GROSS_IMAGE_PATH1
    FROM jlyxz.PIC_MATCHTASK
    WHERE GROSS_WEIGH_TIME IS NOT NULL
    """

    if last_gross_time is None:
        where_clause = ""
        params = {}
    else:
        where_clause = """
        AND (
              GROSS_WEIGH_TIME > :last_time
           OR (GROSS_WEIGH_TIME = :last_time AND TASK_ID > :last_task_id)
        )
        """
        params = {
            'last_time': datetime.strptime(last_gross_time, '%Y-%m-%d %H:%M:%S'),
            'last_task_id': last_task_id or '0',
        }

    order_clause = "ORDER BY GROSS_WEIGH_TIME, TASK_ID"
    sql = "\n".join([base_sql, where_clause, order_clause])

    cursor = connection.cursor()
    cursor.execute(sql, **params)
    columns = [c[0] for c in cursor.description]
    data = cursor.fetchall()
    cursor.close()

    if not data:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(data, columns=columns)


class AutoSiamesePredictor:
    """自动轮询处理增量记录，并输出疑似套牌结果到 CSV"""

    def __init__(self):
        self.model = Siamese()
        self.cropper = VehicleCropper()

    def _ensure_and_get_local_path(self, remote_path):
        """保证远程路径对应的本地文件存在，必要时请求服务器下载，返回本地路径或 None"""
        if not remote_path or not isinstance(remote_path, str):
            return None
        remote_path = remote_path.strip()
        if not remote_path:
            return None

        local_path = ensure_local_path(remote_path)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path

        # 本地没有，则尝试从服务器拉取
        save_dir = os.path.dirname(local_path)
        os.makedirs(save_dir, exist_ok=True)

        print(f"请求服务器文件: {remote_path}")
        print(f"本地保存到: {local_path}")
        success = request_image(SERVER_IP, SERVER_PORT, remote_path, save_dir)
        if not success:
            print(f"下载失败: {remote_path}")
            return None
        return local_path if os.path.exists(local_path) and os.path.getsize(local_path) > 0 else None

    def _compare_two_images(self, path1, path2):
        """比对两张图片，相似度小于等于 0.3 视为疑似套牌，返回概率或 None"""
        if not path1 or not path2:
            return None
        if not isinstance(path1, str) or not isinstance(path2, str):
            return None
        path1 = path1.strip()
        path2 = path2.strip()
        if not path1 or not path2:
            return None
        if (not os.path.exists(path1)) or (not os.path.exists(path2)):
            return None
        try:
            from PIL import Image

            img1 = Image.open(path1)
            img2 = Image.open(path2)
            img1 = self.cropper.process_pil(img1)
            img2 = self.cropper.process_pil(img2)
            prob = self.model.detect_image(img1, img2)
            prob = prob.item() if hasattr(prob, 'item') else float(prob)
            return prob
        except Exception as e:
            print(f"批量比对时加载或识别图片出错: {e}")
            return None

    def process_new_rows(self, df):
        """对增量记录按车牌顺序进行比对，返回本轮疑似记录列表和最新进度"""
        if df.empty:
            return [], None, None

        df = df.sort_values(['GROSS_WEIGH_TIME', 'TASK_ID'])

        last_record_by_plate = {}
        suspicious_records = []

        latest_gross_time_str = None
        latest_task_id = None

        for _, row in df.iterrows():
            gross_time = row['GROSS_WEIGH_TIME']
            task_id = row['TASK_ID']
            plate = row['TRUCK_ID']

            if plate is None or task_id is None or gross_time is None:
                continue

            plate_str = str(plate).strip()
            if not plate_str:
                continue

            # 更新最新进度
            if isinstance(gross_time, datetime):
                gross_time_str = gross_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                gross_time_str = str(gross_time)
            latest_gross_time_str = gross_time_str
            latest_task_id = str(task_id)

            prev_row = last_record_by_plate.get(plate_str)
            if prev_row is not None:
                suspicious = False

                # 皮重图片比对
                curr_tare_remote = row.get('TARE_IMAGE_PATH1')
                prev_tare_remote = prev_row.get('TARE_IMAGE_PATH1')

                # 确保本地图片存在，本地没有则从服务器拉取
                curr_tare_local = self._ensure_and_get_local_path(curr_tare_remote)
                prev_tare_local = self._ensure_and_get_local_path(prev_tare_remote)

                tare_prob = self._compare_two_images(curr_tare_local, prev_tare_local)
                if tare_prob is not None and tare_prob <= 0.3:
                    suspicious = True
                    suspicious_records.append({
                        'type': 'tare',
                        'task_id': task_id,
                        'prev_task_id': prev_row.get('TASK_ID'),
                        'truck_id': plate_str,
                        'gross_weigh_time': gross_time_str,
                        'curr_path': curr_tare_remote,
                        'prev_path': prev_tare_remote,
                        'probability': tare_prob,
                    })

                # 毛重图片比对
                curr_gross_remote = row.get('GROSS_IMAGE_PATH1')
                prev_gross_remote = prev_row.get('GROSS_IMAGE_PATH1')

                curr_gross_local = self._ensure_and_get_local_path(curr_gross_remote)
                prev_gross_local = self._ensure_and_get_local_path(prev_gross_remote)

                gross_prob = self._compare_two_images(curr_gross_local, prev_gross_local)
                if gross_prob is not None and gross_prob <= 0.3:
                    suspicious = True
                    suspicious_records.append({
                        'type': 'gross',
                        'task_id': task_id,
                        'prev_task_id': prev_row.get('TASK_ID'),
                        'truck_id': plate_str,
                        'gross_weigh_time': gross_time_str,
                        'curr_path': curr_gross_remote,
                        'prev_path': prev_gross_remote,
                        'probability': gross_prob,
                    })

                if suspicious:
                    print(f"疑似套牌: 车牌={plate_str}, TASK_ID={task_id}, 上一条 TASK_ID={prev_row.get('TASK_ID')}")

            # 更新当前车牌的最新记录
            last_record_by_plate[plate_str] = row

        return suspicious_records, latest_gross_time_str, latest_task_id


def append_suspicious_to_csv(records):
    """将疑似记录追加写入自动 CSV 文件"""
    if not records:
        return

    df = pd.DataFrame(records)
    file_exists = os.path.exists(AUTO_CSV_FILE)

    if file_exists:
        df.to_csv(AUTO_CSV_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(AUTO_CSV_FILE, mode='w', header=True, index=False, encoding='utf-8-sig')

    print(f"本轮新增疑似记录 {len(records)} 条，已写入 {AUTO_CSV_FILE}")


def run_once():
    """执行一次从数据库增量拉取并检测疑似套牌车辆的流程"""
    last_gross_time, last_task_id = load_progress()
    print(f"当前进度: last_gross_time={last_gross_time}, last_task_id={last_task_id}")

    connection = connect_to_oracle()
    if not connection:
        print("无法连接到数据库，本次轮询结束")
        return

    try:
        df = fetch_new_rows(connection, last_gross_time, last_task_id)
        if df.empty:
            print("没有新的记录需要处理")
            return

        print(f"本次需处理 {len(df)} 条记录")

        predictor = AutoSiamesePredictor()
        suspicious_records, latest_gross_time_str, latest_task_id = predictor.process_new_rows(df)

        if suspicious_records:
            append_suspicious_to_csv(suspicious_records)
        else:
            print("本轮未发现新的疑似套牌记录")

        if latest_gross_time_str and latest_task_id:
            save_progress(latest_gross_time_str, latest_task_id)
            print(f"更新进度: last_gross_time={latest_gross_time_str}, last_task_id={latest_task_id}")

        print("本次处理完成")

    finally:
        connection.close()


def main(poll_interval_seconds: int = 60):
    """每隔 poll_interval_seconds 秒轮询一次数据库，自动检测疑似套牌车辆"""
    print(f"启动自动套牌检测服务，每 {poll_interval_seconds} 秒检查一次新数据...")
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
