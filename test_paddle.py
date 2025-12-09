# 安装依赖
# pip install paddlepaddle-gpu
# pip install paddlenlp==2.7.0

import time

# 临时补丁：修复 aistudio_sdk.hub 缺少 download 函数的问题
import sys
from unittest.mock import Mock

# 创建一个 mock 的 download 函数
mock_download = Mock(return_value=None)

# 在导入 paddlenlp 之前，将 mock 函数注入到 aistudio_sdk.hub
import aistudio_sdk.hub
aistudio_sdk.hub.download = mock_download

from paddlenlp import Taskflow


print("="*100)
print("PaddleNLP UIE 中文信息抽取测试")
print("="*100)


# 测试案例 1: 基础测试
print("\n" + "="*100)
print("测试案例 1: 基础测试")
print("="*100)

# 定义你想抽的"字段"，相当于自定义实体类型
schema = ["地名"]

# 创建一个信息抽取任务
print("\n正在初始化 PaddleNLP UIE 模型...")
ie = Taskflow("information_extraction", schema=schema)
print("模型加载完成！")

# 输入中文文本
text = "三清山天气"

print(f"\n文本: {text}")
print(f"Schema: {schema}\n")

# 执行信息抽取
result = ie(text)

print("识别结果:")
print(result)
print()

"""
# 测试案例 2: 商业场景
print("="*100)
print("测试案例 2: 商业场景")
print("="*100)

schema = ["人物", "公司", "产品"]
ie = Taskflow("information_extraction", schema=schema)

text = "马云创办的阿里巴巴集团旗下有淘宝、天猫、支付宝等多个知名产品，马化腾创办的腾讯公司也推出了微信、QQ等社交软件。"

print(f"\n文本: {text}")
print(f"Schema: {schema}\n")

result = ie(text)

print("识别结果:")
print(result)
print()


# 测试案例 3: 嵌套关系抽取
print("="*100)
print("测试案例 3: 嵌套关系抽取（合同信息）")
print("="*100)

schema = [
    "人物",
    "公司",
    "地点",
    {"合同": ["合同编号", "签署日期"]}  # 嵌套子字段
]
ie = Taskflow("information_extraction", schema=schema)

text = "2023年5月1日，张三与阿里巴巴在杭州签署了编号为HT-2023-001的采购合同。"

print(f"\n文本: {text}")
print(f"Schema: {schema}\n")

result = ie(text)

print("识别结果:")
print(result)
print()


# 测试案例 4: 扩展实体类型（地点、时间）
print("="*100)
print("测试案例 4: 扩展实体类型（地点、时间）")
print("="*100)

schema = ["人物", "地点", "公司", "产品", "时间"]
ie = Taskflow("information_extraction", schema=schema)

text = "2023年1月，李明在上海参加了华为公司举办的鸿蒙系统发布会。"

print(f"\n文本: {text}")
print(f"Schema: {schema}\n")

result = ie(text)

print("识别结果:")
print(result)
print()


# 测试案例 5: 路径规划场景
print("="*100)
print("测试案例 5: 路径规划场景")
print("="*100)

schema = ["起点", "终点", "地点", "交通方式", "时长"]
ie = Taskflow("information_extraction", schema=schema)

text = "我想从北京开车到上海，途径南京，预计需要12小时。"

print(f"\n文本: {text}")
print(f"Schema: {schema}\n")

result = ie(text)

print("识别结果:")
print(result)
print()


# 测试案例 6: 事件抽取
print("="*100)
print("测试案例 6: 事件抽取")
print("="*100)

schema = {
    "地震触发词": ["地震强度", "时间", "震中位置", "震源深度"]
}
ie = Taskflow("information_extraction", schema=schema)

text = "中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。"

print(f"\n文本: {text}")
print(f"Schema: {schema}\n")

result = ie(text)

print("识别结果:")
print(result)
print()


# 测试案例 7: 复杂关系抽取
print("="*100)
print("测试案例 7: 复杂关系抽取（竞赛信息）")
print("="*100)

schema = {
    "竞赛名称": ["主办方", "承办方", "已举办次数"]
}
ie = Taskflow("information_extraction", schema=schema)

text = "2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会承办，已连续举办4届。"

print(f"\n文本: {text}")
print(f"Schema: {schema}\n")

result = ie(text)

print("识别结果:")
print(result)
print()


# 性能测试
print("="*100)
print("性能测试: 处理不同长度的文本")
print("="*100)

test_cases = [
    {
        "text": "张三去北京。",
        "schema": ["人物", "地点"]
    },
    {
        "text": "昨天李四在上海和腾讯公司的同事讨论了微信的新功能。",
        "schema": ["人物", "地点", "公司", "产品"]
    },
    {
        "text": "2023年，王五在深圳创办了一家科技公司，主要开发人工智能产品。公司团队包括来自百度、阿里巴巴的资深工程师。他们的第一款产品是一个智能客服系统，已经在多个电商平台上线使用。",
        "schema": ["人物", "地点", "公司", "产品", "时间"]
    }
]

for i, test_case in enumerate(test_cases, 1):
    text = test_case["text"]
    schema = test_case["schema"]
    
    # 创建任务实例
    ie = Taskflow("information_extraction", schema=schema)
    
    start_time = time.time()
    result = ie(text)
    elapsed_time = time.time() - start_time
    
    print(f"\n文本 {i} (长度: {len(text)} 字):")
    print(f"  内容: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"  Schema: {schema}")
    print(f"  耗时: {elapsed_time:.4f} 秒")
    print(f"  结果: {result}")
"""

print("\n" + "="*100)
print("测试完成！")
print("="*100)

