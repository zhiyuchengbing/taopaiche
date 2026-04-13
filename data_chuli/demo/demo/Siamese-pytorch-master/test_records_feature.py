"""
测试记录查询功能
运行此脚本前，请确保服务已启动：python my_predict_gui_new2.py
"""
import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8001"

def test_query_records():
    """测试查询记录"""
    print("测试查询记录...")
    
    # 计算日期范围
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "case_type": "all",
        "limit": 10,
        "offset": 0
    }
    
    response = requests.get(f"{BASE_URL}/api/records", params=params)
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"总记录数: {data.get('total', 0)}")
        print(f"返回记录数: {len(data.get('records', []))}")
        
        if data.get('records'):
            print("\n第一条记录:")
            print(json.dumps(data['records'][0], indent=2, ensure_ascii=False))
    else:
        print(f"错误: {response.text}")

def test_get_record(record_id):
    """测试获取单条记录"""
    print(f"\n测试获取记录 {record_id}...")
    
    response = requests.get(f"{BASE_URL}/api/record/{record_id}")
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(f"错误: {response.text}")

def test_protect_record(record_id):
    """测试保护记录"""
    print(f"\n测试保护记录 {record_id}...")
    
    payload = {
        "protected": True,
        "note": "测试保护功能"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/record/{record_id}/protect",
        json=payload
    )
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")

if __name__ == "__main__":
    print("=" * 50)
    print("记录查询功能测试")
    print("=" * 50)
    
    # 测试查询
    test_query_records()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)
    print("\n请访问以下页面查看完整功能:")
    print(f"  - 记录查询: {BASE_URL}/records")
    print(f"  - 预测页面: {BASE_URL}/ui")
    print(f"  - 运行统计: {BASE_URL}/dashboard")
