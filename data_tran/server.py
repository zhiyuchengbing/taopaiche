import socket
import os
import sys
import time
import json
from datetime import datetime

# 配置参数
RESTART_DELAY_SECONDS = 10  # 重启前等待的秒数
MAX_NO_SEND_HOURS = 24  # 超过多少小时没有发送数据就重启（默认24小时）

# 脚本所在目录，保证无论从哪里运行，进度文件路径都是固定的
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRESS_FILE = os.path.join(BASE_DIR, 'server_progress.json')


def load_last_send_time():
    """加载最后一次发送数据的时间"""
    if not os.path.exists(PROGRESS_FILE):
        return None
    try:
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('last_send_time')
    except Exception:
        return None


def save_last_send_time():
    """保存最后一次发送数据的时间"""
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {
        'last_send_time': current_time_str,
    }
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def check_need_restart():
    """检查是否需要重启（长时间没有发送数据）"""
    last_send_time = load_last_send_time()
    if last_send_time is None:
        # 如果从未发送过，不重启（可能是刚启动）
        return False
    
    try:
        last_send = datetime.strptime(last_send_time, '%Y-%m-%d %H:%M:%S')
        now = datetime.now()
        hours_since_send = (now - last_send).total_seconds() / 3600
        
        if hours_since_send >= MAX_NO_SEND_HOURS:
            print(f"\n警告: 已经 {hours_since_send:.2f} 小时没有向客户端发送任何数据")
            print(f"超过阈值 {MAX_NO_SEND_HOURS} 小时，将重启服务器...")
            return True
    except Exception as e:
        print(f"检查发送时间时出错: {e}")
    
    return False


def check_file_readable(file_path):
    """检查服务器上文件是否存在且可读"""
    if not os.path.isfile(file_path):
        print(f"文件不存在或不是普通文件: {file_path}")
        return False
    if not os.access(file_path, os.R_OK):
        print(f"没有读取权限: {file_path}")
        return False
    return True


def send_image(conn, file_path):
    """从服务器读取文件并发送给客户端"""
    try:
        # 先检查文件可读性，如果不行，发一个明显的空响应给客户端
        if not check_file_readable(file_path):
            # 文件名长度 0
            conn.sendall((0).to_bytes(4, byteorder="big"))
            # 文件大小 0
            conn.sendall((0).to_bytes(8, byteorder="big"))
            return False

        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        # 先发送文件名长度和文件名
        conn.sendall(len(file_name).to_bytes(4, byteorder="big"))
        conn.sendall(file_name.encode("utf-8"))

        # 再发送文件大小（8 字节）
        conn.sendall(file_size.to_bytes(8, byteorder="big"))

        print(f"正在发送文件: {file_path} ({file_size} 字节)")

        # 发送文件内容
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                conn.sendall(chunk)

        print("文件发送完成")
        # 成功发送后，更新最后发送时间
        save_last_send_time()
        return True

    except Exception as e:
        print(f"发送文件时出错: {e}")
        return False


def start_server(host="0.0.0.0", port=5000):
    """启动文件/图片发送服务器，客户端发送完整文件路径"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(5)

        print(f"图片服务器已启动，监听 {host}:{port}")
        print("客户端需要发送服务器上的 *完整文件路径*")

        print("按 Ctrl+C 停止服务器\n")

        try:
            while True:
                # 检查是否需要重启（长时间无发送）
                if check_need_restart():
                    print(f"\n等待 {RESTART_DELAY_SECONDS} 秒后重启...")
                    time.sleep(RESTART_DELAY_SECONDS)
                    raise Exception("长时间未发送数据，触发重启")
                
                conn, addr = s.accept()
                print(f"\n[+] 客户端已连接: {addr}")

                try:
                    # 客户端先发 4 字节“路径字符串”长度
                    name_len_bytes = conn.recv(4)

                    if not name_len_bytes:
                        print("未收到文件名长度，关闭连接")
                        conn.close()
                        continue

                    path_length = int.from_bytes(name_len_bytes, byteorder="big")
                    remote_path = conn.recv(path_length).decode("utf-8")

                    # 直接把客户端发来的完整路径当作服务器本地路径
                    file_path = remote_path
                    print(f"客户端请求文件: {file_path}")

                    if send_image(conn, file_path):

                        print("文件发送成功")
                    else:
                        print("文件发送失败")

                except Exception as e:
                    print(f"处理客户端请求时出错: {e}")
                finally:
                    conn.close()
                    print(f"[-] 客户端已断开: {addr}")

        except KeyboardInterrupt:
            print("\n服务器正在关闭...")
            raise  # 重新抛出 KeyboardInterrupt，让主程序正常退出
        except Exception as e:
            print(f"服务器错误: {e}")
            import traceback
            traceback.print_exc()
            raise  # 重新抛出异常，让主程序捕获并重启


def restart_program():
    """重启程序"""
    print("\n" + "=" * 60)
    print("正在重启服务器...")
    print("=" * 60 + "\n")
    python = sys.executable
    os.execl(python, python, *sys.argv)


if __name__ == "__main__":
    # 客户端会发送服务器上的完整文件路径，这里只需要启动监听即可
    print("启动图片服务器（支持自动重启）...")
    print(f"配置: 超过 {MAX_NO_SEND_HOURS} 小时无发送将自动重启")
    
    try:
        while True:
            try:
                start_server(port=5000)
            except KeyboardInterrupt:
                print("\n收到中断信号，服务器正常退出。")
                break
            except Exception as e:
                print(f"\n服务器发生异常: {e}")
                import traceback
                traceback.print_exc()
                print(f"\n等待 {RESTART_DELAY_SECONDS} 秒后自动重启服务器...")
                time.sleep(RESTART_DELAY_SECONDS)
                restart_program()
    except KeyboardInterrupt:
        print("\n收到中断信号，服务器退出。")