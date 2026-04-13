"""
测试复核和导出功能
"""
import requests
import json

BASE_URL = "http://localhost:8001"

def test_review_workflow():
    """测试复核工作流程"""
    print("=" * 60)
    print("测试复核功能")
    print("=" * 60)
    
    # 1. 查询记录
    print("\n1. 查询记录...")
    response = requests.get(f"{BASE_URL}/api/records", params={
        "limit": 5,
        "case_type": "fake_plate"
    })
    data = response.json()
    
    if not data.get("records"):
        print("❌ 没有找到记录")
        return
    
    record_id = data["records"][0]["record_id"]
    print(f"✅ 找到记录: {record_id}")
    
    # 2. 提交复核
    print("\n2. 提交复核...")
    response = requests.post(f"{BASE_URL}/api/record/{record_id}/review", json={
        "reviewed_case_type": "normal",
        "review_reason": "经人工核实，两车为同一车辆，系统误判",
        "reviewed_by": "测试人员",
        "review_confidence": "high"
    })
    data = response.json()
    
    if data.get("ok"):
        print(f"✅ 复核成功: {data.get('message')}")
    else:
        print(f"❌ 复核失败: {data.get('error')}")
        return
    
    # 3. 查看复核结果
    print("\n3. 查看复核结果...")
    response = requests.get(f"{BASE_URL}/api/record/{record_id}")
    record = response.json()
    
    print(f"   系统判定: {record.get('case_type')}")
    print(f"   复核结果: {record.get('reviewed_case_type')}")
    print(f"   复核人员: {record.get('reviewed_by')}")
    print(f"   复核理由: {record.get('review_reason')}")
    
    # 4. 获取复核统计
    print("\n4. 获取复核统计...")
    response = requests.get(f"{BASE_URL}/api/records/review_stats")
    stats = response.json()
    
    print(f"   总记录数: {stats.get('total_records')}")
    print(f"   已复核: {stats.get('reviewed_count')}")
    print(f"   复核率: {stats.get('review_rate', 0) * 100:.1f}%")
    print(f"   确认: {stats.get('accuracy', {}).get('confirmed')}")
    print(f"   修正: {stats.get('accuracy', {}).get('corrected')}")
    
    # 5. 撤销复核
    print("\n5. 撤销复核...")
    response = requests.delete(f"{BASE_URL}/api/record/{record_id}/review")
    data = response.json()
    
    if data.get("ok"):
        print(f"✅ 撤销成功: {data.get('message')}")
    else:
        print(f"❌ 撤销失败: {data.get('error')}")


def test_export_workflow():
    """测试导出工作流程"""
    print("\n" + "=" * 60)
    print("测试导出功能")
    print("=" * 60)
    
    # 1. 查询记录
    print("\n1. 查询异常记录...")
    response = requests.get(f"{BASE_URL}/api/records", params={
        "limit": 3,
        "case_type": "all"
    })
    data = response.json()
    
    if not data.get("records"):
        print("❌ 没有找到记录")
        return
    
    records = [r for r in data["records"] if r.get("case_type") in ["fake_plate", "change_trailer"]]
    if not records:
        print("❌ 没有找到异常记录")
        return
    
    record_ids = [r["record_id"] for r in records[:3]]
    print(f"✅ 找到 {len(record_ids)} 条异常记录")
    
    # 2. 导出单条记录
    print("\n2. 导出单条记录...")
    response = requests.post(f"{BASE_URL}/api/record/{record_ids[0]}/export", json={
        "include_meta": True
    })
    data = response.json()
    
    if data.get("ok"):
        print(f"✅ 导出成功")
        print(f"   路径: {data.get('export_path')}")
        print(f"   消息: {data.get('message')}")
    else:
        print(f"❌ 导出失败: {data.get('error')}")
    
    # 3. 批量导出
    if len(record_ids) > 1:
        print("\n3. 批量导出...")
        response = requests.post(f"{BASE_URL}/api/records/batch_export", json={
            "record_ids": record_ids,
            "group_by": "case_type",
            "include_summary": True
        })
        data = response.json()
        
        if data.get("ok"):
            print(f"✅ 批量导出成功")
            print(f"   路径: {data.get('export_path')}")
            print(f"   消息: {data.get('message')}")
            print(f"   记录数: {data.get('total_records')}")
        else:
            print(f"❌ 批量导出失败: {data.get('error')}")
    
    # 4. 按条件导出
    print("\n4. 按条件导出...")
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    response = requests.post(f"{BASE_URL}/api/records/export_by_filter", json={
        "start_date": start_date,
        "end_date": end_date,
        "case_types": ["fake_plate", "change_trailer"]
    })
    data = response.json()
    
    if data.get("ok"):
        print(f"✅ 按条件导出成功")
        print(f"   路径: {data.get('export_path')}")
        print(f"   消息: {data.get('message')}")
    else:
        print(f"❌ 按条件导出失败: {data.get('error')}")


def main():
    print("\n🚀 开始测试复核和导出功能\n")
    
    # 测试健康检查
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ 服务正常运行\n")
        else:
            print("❌ 服务异常")
            return
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        print(f"   请确保服务已启动: python my_predict_gui_new2.py")
        return
    
    # 运行测试
    test_review_workflow()
    test_export_workflow()
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
    print("\n访问以下页面查看效果：")
    print(f"  - 记录查询: {BASE_URL}/records")
    print(f"  - 复核统计: {BASE_URL}/review_stats")
    print(f"  - 运行统计: {BASE_URL}/dashboard")
    print()


if __name__ == "__main__":
    main()
