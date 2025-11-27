from huggingface_hub import InferenceClient

from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-v2-m3')


# 设置查询
query1 = "服用阿利沙坦酯氨氯地平后血"

# 设置文档集合（直接粘贴多行文本即可）
documents_text = """
服用阿利沙坦酯氨氯地平后血压总在正常边缘
阿利沙坦酯氨氯地平服用时间
服用阿利沙坦酯氨氯地平片运动会让血压升高
阿利沙坦酯氨氯地平片怎么服用
阿利沙坦酯氨氯地平片服用时间
服用阿利沙坦酯氨氯地平片三个月之后可以停用吗
阿利沙坦酯氨氯地平片服用方法
阿利沙坦酯氨氯地平片过量服用
阿利沙坦酯氨氯地平片副作用
阿利沙坦酯氨氯地平片如何服用
"""

# 自动处理成列表（去除空行）
documents = [line.strip() for line in documents_text.strip().split('\n') if line.strip()]

# 批量计算分数
results = []
for doc in documents:
    score = reranker.compute_score([query1, doc], normalize=True)
    results.append((query1, doc, score[0]))

# 按分数从高到低排序
results.sort(key=lambda x: x[2], reverse=True)

# 输出排序后的结果
for query, doc, score in results:
    print(f"{query:<30}{doc:<40}{score:.4f}")
