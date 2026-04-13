import os
import cx_Oracle
import pandas as pd
from datetime import datetime, timedelta

def connect_to_oracle():
    try:
        # 设置 Oracle Instant Client 路径
        os.environ["PATH"] = r"D:\instantclient-basic-windows.x64-23.26.0.0.0\instantclient_23_0" + ";" + os.environ["PATH"]
        os.environ["TNS_ADMIN"] = r"D:\instantclient-basic-windows.x64-23.26.0.0.0\instantclient_23_0\network\admin"
        
        # 创建数据库连接
        dsn_tns = cx_Oracle.makedsn('10.100.2.229', '1521', service_name='JLYXZ')
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