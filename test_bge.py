from FlagEmbedding import FlagReranker, FlagLLMReranker

# ===========================================
# 模型配置 - 修改这里切换不同模型
# ===========================================
# 可选模型:
#   普通 reranker (使用 FlagReranker):
#     - 'BAAI/bge-reranker-base'        (轻量级，中英文)
#     - 'BAAI/bge-reranker-large'       (轻量级，中英文)
#     - 'BAAI/bge-reranker-v2-m3'       (多语言，推荐)
#   LLM-based reranker (使用 FlagLLMReranker):
#     - 'BAAI/bge-reranker-v2-gemma'    (基于gemma-2b，多语言，效果更好)

# 要对比的模型列表
MODELS_TO_COMPARE = [
    # 轻量级模型 - 推理快，适合部署
    {'name': 'BAAI/bge-reranker-base', 'type': 'normal', 'desc': '基础版-轻量快速'},
    {'name': 'BAAI/bge-reranker-large', 'type': 'normal', 'desc': '大版本-平衡性能'},
    
    # 多语言模型 - 性能强，支持多语言
    {'name': 'BAAI/bge-reranker-v2-m3', 'type': 'normal', 'desc': '多语言-推荐'},
    {'name': 'BAAI/bge-reranker-v2-gemma', 'type': 'llm', 'desc': 'LLM增强-高性能'},
]

# 可以根据需求注释掉不需要测试的模型，例如：
# MODELS_TO_COMPARE = [
#     {'name': 'BAAI/bge-reranker-v2-m3', 'type': 'normal', 'desc': '多语言'},
#     {'name': 'BAAI/bge-reranker-v2-gemma', 'type': 'llm', 'desc': 'LLM增强'},
# ]

# 设置查询
query1 = "b站股票"

# 设置文档集合（直接粘贴多行文本即可）
documents_text = """
哔哩哔哩股票
b站视频
"""

# 自动处理成列表（去除空行）
documents = [line.strip() for line in documents_text.strip().split('\n') if line.strip()]


def load_reranker(model_config):
    """根据模型类型加载对应的 reranker"""
    model_name = model_config['name']
    model_type = model_config['type']
    
    if model_type == 'llm':
        # LLM-based reranker，使用 fp16 加速
        return FlagLLMReranker(model_name, use_fp16=True)
    else:
        # 普通 reranker
        return FlagReranker(model_name, use_fp16=True)


def evaluate_model(reranker, query, documents):
    """计算模型对文档的评分"""
    results = []
    for doc in documents:
        score = reranker.compute_score([query, doc], normalize=True)
        # 处理返回值（可能是列表或单个值）
        if isinstance(score, list):
            score = score[0]
        results.append((doc, score))
    
    # 按分数从高到低排序
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# 对比所有模型
print("=" * 80)
print(f"查询: {query1}")
print(f"文档数量: {len(documents)}")
print("=" * 80)

all_model_results = {}

for idx, model_config in enumerate(MODELS_TO_COMPARE, 1):
    model_name = model_config['name']
    model_desc = model_config.get('desc', '')
    print(f"\n[{idx}/{len(MODELS_TO_COMPARE)}] 正在加载模型: {model_name}")
    if model_desc:
        print(f"    描述: {model_desc}")
    
    try:
        reranker = load_reranker(model_config)
        results = evaluate_model(reranker, query1, documents)
        all_model_results[model_name] = results
        
        print(f"\n【{model_name}】结果:")
        print("-" * 60)
        print(f"{'排名':<6}{'文档':<40}{'分数':<10}")
        print("-" * 60)
        for rank, (doc, score) in enumerate(results, 1):
            print(f"{rank:<6}{doc:<40}{score:.6f}")
    except Exception as e:
        print(f"    ❌ 模型加载或推理失败: {e}")
        print(f"    提示: 如果是首次使用该模型，会自动下载，请确保网络连接正常")

# 打印对比总结
if all_model_results:
    print("\n" + "=" * 80)
    print("模型对比总结")
    print("=" * 80)
    print(f"{'文档':<30}", end="")
    for model_name in all_model_results.keys():
        short_name = model_name.split('/')[-1][:20]  # 限制长度
        print(f"{short_name:<25}", end="")
    print()
    print("-" * 80)

    for doc in documents:
        print(f"{doc:<30}", end="")
        for model_name, results in all_model_results.items():
            score = next(s for d, s in results if d == doc)
            print(f"{score:<25.6f}", end="")
        print()
    
    print("\n" + "=" * 80)
    print("模型推荐:")
    print("-" * 80)
    print("• 如果追求速度: bge-reranker-base (最快)")
    print("• 如果追求平衡: bge-reranker-v2-m3 (推荐)")
    print("• 如果追求效果: bge-reranker-v2-gemma (最准确)")
    print("=" * 80)
else:
    print("\n⚠️ 没有成功加载任何模型，请检查网络连接或模型配置")
