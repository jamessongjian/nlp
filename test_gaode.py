import requests
import time

def test_route_planning_performance(start_location, end_location, amap_key):
    """
    测试路径规划功能的整体耗时
    
    参数:
        start_location (str): 起点地址，如"北京"
        end_location (str): 终点地址，如"上海"
        amap_key (str): 高德地图API Key
        
    返回:
        dict: 包含耗时信息和结果的字典
    """
    print("="*100)
    print("路径规划性能测试")
    print("="*100)
    
    # 记录开始时间
    total_start_time = time.time()
    
    result = {
        "start_location": start_location,
        "end_location": end_location,
        "geocode_time": 0,
        "route_planning_time": 0,
        "total_time": 0,
        "success": False,
        "error": None
    }
    
    try:
        # 步骤1: 获取起点经纬度
        print(f"\n1. 正在获取起点 '{start_location}' 的经纬度...")
        geocode_start_time = time.time()
        
        geocode_url = "https://restapi.amap.com/v3/geocode/geo"
        params = {
            "address": start_location,
            "key": amap_key,
            "output": "json"
        }
        
        response = requests.get(geocode_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "1" and data.get("geocodes"):
            start_coords = data["geocodes"][0]["location"]
            print(f"   起点经纬度: {start_coords}")
        else:
            raise Exception(f"获取起点坐标失败: {data.get('info', '未知错误')}")
        
        # 步骤2: 获取终点经纬度
        print(f"\n2. 正在获取终点 '{end_location}' 的经纬度...")
        
        params["address"] = end_location
        response = requests.get(geocode_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "1" and data.get("geocodes"):
            end_coords = data["geocodes"][0]["location"]
            print(f"   终点经纬度: {end_coords}")
        else:
            raise Exception(f"获取终点坐标失败: {data.get('info', '未知错误')}")
        
        geocode_end_time = time.time()
        result["geocode_time"] = geocode_end_time - geocode_start_time
        print(f"\n   地理编码总耗时: {result['geocode_time']:.3f} 秒")
        
        # 步骤3: 调用驾车路线规划API
        print(f"\n3. 正在规划从 '{start_location}' 到 '{end_location}' 的驾车路线...")
        route_start_time = time.time()
        
        route_url = "https://restapi.amap.com/v5/direction/driving"
        route_params = {
            "origin": start_coords,
            "destination": end_coords,
            "key": amap_key,
            "output": "json"
        }
        
        response = requests.get(route_url, params=route_params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        route_end_time = time.time()
        result["route_planning_time"] = route_end_time - route_start_time
        
        if data.get("status") == "1" and data.get("route"):
            paths = data["route"].get("paths", [])
            if paths:
                path = paths[0]
                distance = float(path.get("distance", 0)) / 1000  # 转换为公里
                duration = float(path.get("duration", 0)) / 60  # 转换为分钟
                
                print(f"   路线规划成功!")
                print(f"   距离: {distance:.2f} 公里")
                print(f"   预计时间: {duration:.0f} 分钟")
                print(f"   路线规划耗时: {result['route_planning_time']:.3f} 秒")
                
                result["distance_km"] = distance
                result["duration_minutes"] = duration
                result["success"] = True
            else:
                raise Exception("未找到可用路线")
        else:
            raise Exception(f"路线规划失败: {data.get('info', '未知错误')}")
            
    except requests.exceptions.Timeout:
        result["error"] = "请求超时"
        print(f"\n   错误: 请求超时")
    except requests.exceptions.RequestException as e:
        result["error"] = f"网络请求错误: {str(e)}"
        print(f"\n   错误: {result['error']}")
    except Exception as e:
        result["error"] = str(e)
        print(f"\n   错误: {result['error']}")
    
    # 计算总耗时
    total_end_time = time.time()
    result["total_time"] = total_end_time - total_start_time
    
    print("\n" + "="*100)
    print("性能测试结果汇总")
    print("="*100)
    print(f"起点: {start_location}")
    print(f"终点: {end_location}")
    print(f"地理编码耗时: {result['geocode_time']:.3f} 秒")
    print(f"路线规划耗时: {result['route_planning_time']:.3f} 秒")
    print(f"总耗时: {result['total_time']:.3f} 秒")
    print(f"状态: {'成功' if result['success'] else '失败'}")
    if result.get("error"):
        print(f"错误信息: {result['error']}")
    print("="*100 + "\n")
    
    return result


# 使用示例
if __name__ == "__main__":
    # 请替换为您自己的高德地图API Key
    # 可以在这里申请: https://console.amap.com/dev/key/app
    AMAP_KEY = "c4e505d63a8dd4c2ea8e9658ad360aa3"  # 请替换为您的Key
    
    # 测试路径规划性能
    result = test_route_planning_performance(
        start_location="北京",
        end_location="上海",
        amap_key=AMAP_KEY
    )
    
    # 可以进行多次测试以获得平均耗时
    print("\n进行3次测试以获得平均耗时...")
    times = []
    for i in range(3):
        print(f"\n第 {i+1} 次测试:")
        result = test_route_planning_performance("北京", "福州西站", AMAP_KEY)
        if result["success"]:
            times.append(result["total_time"])
        time.sleep(1)  # 避免请求过于频繁
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n平均耗时: {avg_time:.3f} 秒")
