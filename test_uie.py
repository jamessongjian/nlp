from transformers import AutoModel, AutoTokenizer

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("xusenlin/uie-base", trust_remote_code=True)
model = AutoModel.from_pretrained("xusenlin/uie-base", trust_remote_code=True)

"""
print("="*100)
print("1. 实体抽取示例")
print("="*100)

schema = ["时间", "选手", "赛事名称"]
text = "2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"
result = model.predict(tokenizer, text, schema=schema)
print(f"文本: {text}")
print(f"Schema: {schema}")
print(f"结果: {result}\n")

print("="*100)
print("2. 关系抽取示例")
print("="*100)

schema = {'竞赛名称': ['主办方', '承办方', '已举办次数']}
text = "2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。"
model.set_schema(schema)
result = model.predict(tokenizer, text)
print(f"文本: {text}")
print(f"Schema: {schema}")
print(f"结果: {result}\n")

print("="*100)
print("3. 事件抽取示例")
print("="*100)

schema = {'地震触发词': ['地震强度', '时间', '震中位置', '震源深度']}
text = "中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。"
model.set_schema(schema)
result = model.predict(tokenizer, text)
print(f"文本: {text}")
print(f"Schema: {schema}")
print(f"结果: {result}\n")

print("="*100)
print("4. 观点抽取示例")
print("="*100)

schema = {'评价维度': ['观点词', '情感倾向[正向，负向]']}
text = "店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队"
model.set_schema(schema)
result = model.predict(tokenizer, text)
print(f"文本: {text}")
print(f"Schema: {schema}")
print(f"结果: {result}\n")

print("="*100)
print("5. 情感分类示例")
print("="*100)

schema = "情感倾向[正向，负向]"
text = "这个产品用起来真的很流畅，我非常喜欢"
model.set_schema(schema)
result = model.predict(tokenizer, text)
print(f"文本: {text}")
print(f"Schema: {schema}")
print(f"结果: {result}\n")

print("="*100)
print("6. 复杂信息抽取示例（法律文书）")
print("="*100)

schema = ['法院', {'原告': '委托代理人'}, {'被告': '委托代理人'}]
text = "北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。"
model.set_schema(schema)
result = model.predict(tokenizer, text)
print(f"文本: {text}")
print(f"Schema: {schema}")
print(f"结果: {result}\n")

print("="*100)
print("7. 自定义测试 - 医药信息抽取")
print("="*100)

schema = ["药品名称", "症状", "副作用"]
text = "服用阿利沙坦酯氨氯地平后血压总在正常边缘，可能会出现头晕等副作用"
model.set_schema(schema)
result = model.predict(tokenizer, text)
print(f"文本: {text}")
print(f"Schema: {schema}")
print(f"结果: {result}\n")

"""
print("="*100)
print("8. SC NER测试 - 股市信息提取")
print("="*100)

schema = ["公司名称", "股市类型[A股，港股，美股]"]
text = "阿里股票"
model.set_schema(schema)
result = model.predict(tokenizer, text)
print(f"文本: {text}")
print(f"Schema: {schema}")
print(f"结果: {result}\n")


