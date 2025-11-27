from gliner import GLiNER
import time
import warnings

# 过滤掉 transformers 库的警告信息
warnings.filterwarnings("ignore", message=".*truncate to max_length.*")
warnings.filterwarnings("ignore", message=".*token indices sequence length.*")


# Initialize GLiNER with the base model
print("正在加载 GLiNER 模型...")
# 使用 gliner-community 版本以避免兼容性问题
model = GLiNER.from_pretrained("gliner-community/gliner_medium-v2.5")
print("模型加载完成！\n")


print("="*100)
print("GLiNER 中文实体识别测试")
print("="*100)



# 测试案例 1: 基础测试
print("\n" + "="*100)
print("测试案例 1: 基础测试")
print("="*100)

text = "昨天张三在北京和阿里巴巴的同事讨论了钉钉的新版功能。"

# Labels for entity prediction
# Most GLiNER models should work best when entity types are in lower case or title case
labels = ["person", "company", "product"]

print(f"\n文本: {text}")
print(f"标签: {labels}\n")

# Perform entity prediction
entities = model.predict_entities(text, labels, threshold=0.5)

# Display predicted entities and their labels
print("识别结果:")
for entity in entities:
    print(f"  {entity['text']} => {entity['label']} (置信度: {entity['score']:.3f})")
print()

print("\n" + "="*100)
print("测试完成！")
print("="*100)

