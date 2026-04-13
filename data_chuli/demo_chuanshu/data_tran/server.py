import socket
import os

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
        except Exception as e:
            print(f"服务器错误: {e}")


if __name__ == "__main__":
    # 客户端会发送服务器上的完整文件路径，这里只需要启动监听即可
    start_server(port=5000)