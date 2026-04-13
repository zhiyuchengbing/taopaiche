import os
import sys
import cx_Oracle
import pandas as pd
from datetime import datetime, timedelta

# Oracle 客户端配置（只在进程内设置一次，避免 PATH 被无限累加）
#
# 这里的配置用于让 cx_Oracle / Oracle Client 在 Windows 下能够找到：
# - OCI 运行库（通过把 Instant Client 目录加入 PATH）
# - tnsnames.ora / sqlnet.ora 等网络配置（通过设置 TNS_ADMIN）
#
# Instant Client 的安装目录（包含 oci.dll 等文件）。
# 注意：
# - 这个路径必须真实存在；否则 cx_Oracle.connect 会报 “DPI-1047: Cannot locate a 64-bit Oracle Client library”。
# - 若你升级/更换 Instant Client 版本，需要同步修改这里。
INSTANTCLIENT_DIR = r"D:\instantclient-basic-windows.x64-23.26.0.0.0\instantclient_23_0"
# Oracle 网络配置目录（相当于把 network/admin 作为 tnsnames.ora 的查找路径）。
# 常见放置文件：
# - tnsnames.ora：TNS 别名配置（如果你用别名连接，而不是直连 host/port/service_name）。
# - sqlnet.ora：网络层配置（如字符集、加密、超时等）。
INSTANTCLIENT_NET = os.path.join(INSTANTCLIENT_DIR, "network", "admin")
_ORACLE_ENV_SET = False


def _ensure_oracle_env():
    """
    确保 Oracle 相关环境变量只设置一次，并对 PATH 去重，防止多次重启导致 PATH 无限变长。
    如果 PATH 已经超过 30000 字符，会主动清理，只保留必要的路径。
    """
    global _ORACLE_ENV_SET
    if _ORACLE_ENV_SET:
        return

    # 获取当前 PATH，如果获取失败或为空，使用空字符串。
    # Windows 环境变量有长度限制（历史上常见限制 32767 字符）。
    # 当 PATH 极长时：
    # - 读取可能失败
    # - 设置可能抛异常
    # 因此这里要做“超长检测 + 清理 + 去重”。
    current_path = ""
    path_too_long = False
    # 留出一定余量，避免逼近系统上限后再次追加导致失败。
    MAX_PATH_LENGTH = 30000
    
    try:
        current_path = os.environ.get("PATH", "")
        # 检查 PATH 长度，如果超过 30000 字符（留 2767 字符余量），需要清理
        path_too_long = len(current_path) > MAX_PATH_LENGTH
    except (OSError, ValueError, OverflowError) as e:
        # 如果获取 PATH 时出错（可能是因为 PATH 太长），直接进入强制清理模式
        print(f"获取 PATH 环境变量时出错: {e}，将使用强制清理模式")
        path_too_long = True
        current_path = ""
    except Exception:
        # 其他未知错误，也使用空字符串
        current_path = ""

    def dedup_path(path_str: str, force_clean=False):
        """
        对 PATH 进行去重处理。
        如果 force_clean=True，则只保留 instantclient 和系统关键路径。
        """
        parts = []
        seen = set()

        def add_part(p: str):
            key = p.lower().strip()
            if p and key and key not in seen:
                seen.add(key)
                parts.append(p.strip())

        # 始终优先添加 Instant Client：
        # - 放在 PATH 前面通常能避免命中其它 Oracle 安装残留导致的版本冲突。
        add_part(INSTANTCLIENT_DIR)

        if force_clean:
            # 强制清理模式：只保留“足以运行脚本”的最小 PATH。
            # 目的：当 PATH 太长导致无法读取/写入时，避免整个进程环境不可用。
            # 这里主要保留：
            # - Windows 系统关键路径（保证基本命令可用）
            # - Python 解释器与 Scripts（保证当前 Python 及其脚本可用）
            system_paths = [
                r"C:\Windows\system32",
                r"C:\Windows",
                r"C:\Windows\System32\Wbem",
            ]
            for sys_path in system_paths:
                if os.path.exists(sys_path):
                    add_part(sys_path)
            
            # 尝试保留 Python 相关路径（避免清理后 python/pip 等不可用）
            python_exe = sys.executable if 'sys' in globals() else None
            if python_exe:
                python_dir = os.path.dirname(python_exe)
                if python_dir:
                    add_part(python_dir)
                    # 添加 Scripts 目录
                    scripts_dir = os.path.join(python_dir, "Scripts")
                    if os.path.exists(scripts_dir):
                        add_part(scripts_dir)
        else:
            # 正常模式：对所有 PATH 项做去重。
            # 注意：Windows 下 PATH 分隔符为 ';'。
            for part in path_str.split(";"):
                if part.strip():
                    add_part(part.strip())

        result = ";".join(parts)
        # 再次检查结果长度，如果还是太长，强制清理
        if len(result) > MAX_PATH_LENGTH:
            return dedup_path("", force_clean=True)
        return result

    try:
        # 得到清理/去重后的 PATH：
        # - path_too_long=True 时走强制清理（只留关键路径）
        # - 否则只做去重（保留原有 PATH 能力）
        cleaned_path = dedup_path(current_path, force_clean=path_too_long)
        
        # 如果当前 PATH 太长，先尝试删除再设置
        if path_too_long or len(current_path) > 30000:
            try:
                # 先删除 PATH（如果存在）
                if "PATH" in os.environ:
                    del os.environ["PATH"]
            except Exception:
                pass  # 忽略删除失败
        
        # 设置新的 PATH（仅影响当前 Python 进程及其子进程，不会永久修改系统环境变量）
        os.environ["PATH"] = cleaned_path
        # 设置 Oracle 网络配置路径（让 Oracle Client 在该目录下查找 tnsnames.ora/sqlnet.ora 等）
        os.environ["TNS_ADMIN"] = INSTANTCLIENT_NET
        _ORACLE_ENV_SET = True
        
        if path_too_long:
            print(f"警告: PATH 环境变量过长 ({len(current_path)} 字符)，已自动清理为 {len(cleaned_path)} 字符")
    except Exception as e:
        # 如果设置环境变量失败，尝试强制清理模式
        print(f"设置 PATH 时出错: {e}，尝试强制清理...")
        try:
            # 先删除 PATH
            try:
                if "PATH" in os.environ:
                    del os.environ["PATH"]
            except Exception:
                pass
            
            cleaned_path = dedup_path("", force_clean=True)
            os.environ["PATH"] = cleaned_path
            os.environ["TNS_ADMIN"] = INSTANTCLIENT_NET
            _ORACLE_ENV_SET = True
            print(f"已强制清理 PATH，当前长度: {len(cleaned_path)} 字符")
        except Exception as e2:
            print(f"强制清理 PATH 也失败: {e2}")
            # 最后一次尝试：使用 putenv（虽然不推荐，但作为最后手段）
            try:
                import ctypes
                from ctypes import wintypes
                kernel32 = ctypes.windll.kernel32
                SetEnvironmentVariableW = kernel32.SetEnvironmentVariableW
                SetEnvironmentVariableW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
                SetEnvironmentVariableW.restype = wintypes.BOOL
                
                cleaned_path = dedup_path("", force_clean=True)
                if SetEnvironmentVariableW("PATH", cleaned_path):
                    os.environ["TNS_ADMIN"] = INSTANTCLIENT_NET
                    _ORACLE_ENV_SET = True
                    print(f"使用 Windows API 设置 PATH 成功，当前长度: {len(cleaned_path)} 字符")
                else:
                    raise Exception("Windows API 设置 PATH 失败")
            except Exception as e3:
                print(f"所有方法都失败: {e3}")
                raise

def connect_to_oracle():
    try:
        # 设置 Oracle 客户端环境（只做一次，避免 PATH 无限制累加）
        _ensure_oracle_env()

        # 创建数据库连接
        # 1) 先拼出 DSN（Data Source Name）：描述要连到哪台 Oracle 以及连哪个服务。
        #    - host: 数据库服务器 IP/域名
        #    - port: 监听端口（默认 1521，需与监听配置一致）
        #    - service_name: 服务名（常见于 CDB/PDB 架构；与 SID 不同）
        #    说明：这里使用的是“直连 host/port/service_name”，不依赖 tnsnames.ora。
        dsn_tns = cx_Oracle.makedsn('10.100.2.229', '1521', service_name='JLYXZ')
        # 2) 使用用户名/密码 + DSN 建立会话。
        #    注意：
        #    - 这里把账号密码写在代码里存在泄露风险；建议改为环境变量/配置文件，并避免提交到版本库。
        #    - connect 成功后需要在使用完毕后 connection.close() 释放连接。
        connection = cx_Oracle.connect(user='identify', password='123456', dsn=dsn_tns)
        print("成功连接到Oracle数据库")
        return connection
    except Exception as e:
        print(f"连接数据库时出错: {e}")
        return None

def export_recent_data(connection, table_name, days=30):
    try:
        cursor = connection.cursor()
        
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 查询最近N天的数据
        query = f"""
        SELECT *
        FROM {table_name}
        WHERE CREATED_TIME >= :start_date
        AND CREATED_TIME < :end_date
        ORDER BY CREATED_TIME DESC
        """
        
        print(f"\n正在查询从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的数据...")
        cursor.execute(query, start_date=start_date, end_date=end_date)
        
        # 获取列名
        columns = [col[0] for col in cursor.description]
        
        # 获取数据并转换为DataFrame
        data = cursor.fetchall()
        if not data:
            print("没有找到符合条件的数据")
            return
        
        df = pd.DataFrame(data, columns=columns)
        
        # 生成CSV文件名
        csv_filename = f"{table_name.split('.')[-1]}_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
        
        # 保存为CSV文件
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"成功导出 {len(df)} 条数据到: {os.path.abspath(csv_filename)}")
        
        # 显示数据预览
        print("\n数据预览：")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"导出数据时出错: {e}")
    finally:
        cursor.close()

def main():
    table_name = "jlyxz.PIC_MATCHTASK"
    connection = connect_to_oracle()
    if connection:
        try:
            # 导出最近30天的数据
            export_recent_data(connection, table_name, days=30)
        finally:
            connection.close()
            print("\n数据库连接已关闭")

if __name__ == "__main__":
    main()