import socket
import os

def receive_image(s, save_dir='.'):
    """接收图片并保存"""
    try:
        # 接收文件名
        file_name_length = int.from_bytes(s.recv(4), byteorder='big')
        if file_name_length == 0:
            print("服务器返回: 文件不存在或无法读取")
            return False
        file_name = s.recv(file_name_length).decode('utf-8')
        
        # 接收文件数据
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
        
        # 保存图片
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, 'wb') as f:
            f.write(received_data)
        
        print(f"图片已保存到: {os.path.abspath(save_path)}")
        return True
        
    except Exception as e:
        print(f"接收图片时出错: {e}")
        return False

def request_image(server_ip, server_port, image_name, save_dir='.'):
    """请求并接收图片"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            print(f"正在连接到服务器 {server_ip}:{server_port}...")
            s.connect((server_ip, server_port))
            print("已连接到服务器")
            
            # 发送请求的图片文件名
            s.sendall(len(image_name).to_bytes(4, byteorder='big'))
            s.sendall(image_name.encode('utf-8'))
            
            # 接收图片
            return receive_image(s, save_dir)
            
        except Exception as e:
            print(f"连接服务器时出错: {e}")
            return False

if __name__ == "__main__":
    # 配置
    SERVER_IP = '10.100.2.229'  # 服务器IP
    SERVER_PORT = 5000          # 服务器端口
    SAVE_DIR = 'received_images'  # 本地保存图片的目录
    
    # 创建保存目录（如果不存在）
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    print("=== 图片下载客户端 ===")
    print(f"服务器: {SERVER_IP}:{SERVER_PORT}")
    print(f"本地保存目录: {os.path.abspath(SAVE_DIR)}")
    print("\n输入 *服务器上* 的图片完整路径（例如: D:/AlarmCaptures/17/1/2025/11/01/selected/1.png）")
    print("输入 'exit' 退出程序\n")
    
    while True:
        remote_path = input("请输入服务器图片完整路径: ").strip()
        if remote_path.lower() == 'exit':
            break
            
        if not remote_path:
            continue
            
        # 仅用于本地保存的文件名
        file_name = os.path.basename(remote_path)
        save_dir = SAVE_DIR
        
        print("\n" + "="*50)
        print(f"正在请求服务器文件: {remote_path}")
        print(f"本地保存目录: {os.path.abspath(save_dir)} (文件名: {file_name})")
        
        if request_image(SERVER_IP, SERVER_PORT, remote_path, save_dir):
            print("图片下载完成！")
        else:
            print("图片下载失败")
        
        print("="*50 + "\n")
    
    print("程序已退出")