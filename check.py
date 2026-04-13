import os
import cx_Oracle
import pandas as pd

def connect_to_oracle():
    try:
        # 设置 Oracle Instant Client 路径
        os.environ["PATH"] = r"D:\instantclient-basic-windows.x64-23.26.0.0.0\instantclient_23_0" + ";" + os.environ["PATH"]
        
        # 设置 TNS_ADMIN 环境变量
        os.environ["TNS_ADMIN"] = r"D:\instantclient-basic-windows.x64-23.26.0.0.0\instantclient_23_0\network\admin"
        
        # 创建数据库连接
        dsn_tns = cx_Oracle.makedsn(
            '10.100.2.229',  # 数据库服务器地址
            '1521',          # 端口号
            service_name='JLYXZ'  # 服务名
        )
        
        connection = cx_Oracle.connect(
            user='identify',      # 用户名
            password='123456',    # 密码
            dsn=dsn_tns
        )
        print("成功连接到Oracle数据库")
        return connection
    except Exception as e:
        print(f"连接数据库时出错: {e}")
        print("请检查：")
        print("1. Instant Client 路径是否正确")
        print("2. 数据库服务器地址、端口和服务名是否正确")
        print("3. 网络连接是否正常")
        return None

def show_table_structure(connection, table_name):
    try:
        cursor = connection.cursor()
        
        # 获取表结构
        cursor.execute(f"SELECT * FROM {table_name} WHERE ROWNUM <= 1")
        
        # 获取列信息
        columns = [col[0] for col in cursor.description]
        column_types = [col[1] for col in cursor.description]
        
        # 获取列注释
        comments = []
        for col in cursor.description:
            try:
                cursor.execute(f"""
                    SELECT comments 
                    FROM user_col_comments 
                    WHERE table_name = :tbl 
                    AND column_name = :col
                """, tbl=table_name.split('.')[-1].upper(), col=col[0].upper())
                comment = cursor.fetchone()
                comments.append(comment[0] if comment else "无注释")
            except:
                comments.append("无法获取注释")
        
        # 创建DataFrame显示表结构
        df_structure = pd.DataFrame({
            '列名': columns,
            '数据类型': [str(t) for t in column_types],
            '注释': comments
        })
        
        # 显示表结构
        print("\n表结构信息：")
        print(df_structure)
        
        # 获取前5行数据
        cursor.execute(f"SELECT * FROM {table_name} WHERE ROWNUM <= 5")
        data = cursor.fetchall()
        
        if data:
            df_preview = pd.DataFrame(data, columns=columns)
            print("\n数据预览（前5行）：")
            print(df_preview)
        
        return df_structure
    except Exception as e:
        print(f"获取表结构时出错: {e}")
        return None
    finally:
        cursor.close()

def main():
    table_name = "jlyxz.PIC_MATCHTASK"
    connection = connect_to_oracle()
    if connection:
        try:
            print(f"正在获取表 {table_name} 的结构...")
            structure = show_table_structure(connection, table_name)
            
            if structure is not None:
                print("\n请查看上方的表结构，确认日期字段后，可以修改代码进行数据导出。")
        finally:
            connection.close()
            print("\n数据库连接已关闭")

if __name__ == "__main__":
    main()