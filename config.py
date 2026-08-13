
# ---------- 【必须修改】大模型 API 配置 ----------

DEEPSEEK_API_KEY = " "

# DeepSeek 接口地址
LLM_BASE_URL = "https://api.deepseek.com/v1"
# 使用的模型名称
LLM_MODEL = "deepseek-chat"
# 生成温度，0=最严谨，1=最有创意
LLM_TEMPERATURE = 0.3

# ---------- 知识库与向量库路径 ----------
# 存放原始文档的文件夹（支持 pdf/txt/docx/md）
KNOWLEDGE_DIR = "./knowledge_base"
# 向量数据库保存路径
VECTOR_DB_PATH = "./vector_db"


# 每个文本块的最大字符数
# 课件建议：块过大引入噪声，块过小割裂信息，500是通用折中值
CHUNK_SIZE = 256
# 相邻块之间的重叠字符数，避免边界信息丢失（滑动窗口思想）
CHUNK_OVERLAP = 50


# 使用中文开源嵌入模型，本地运行，免费

EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
   # "BAAI/bge-large-zh-v1.5"

# 运行设备，没有显卡就用 cpu，有显卡改 "cuda"
EMBEDDING_DEVICE = "cpu"
# 是否对向量做归一化（课件提到归一化后余弦相似度计算更准确）
EMBEDDING_NORMALIZE = True

# ---------- 检索配置（课件《向量库与检索详解》章节） ----------
# 检索返回最相关的前 K 个文本片段
# 课件建议：K 太小信息不全，K 太大引入噪声，3-5 为宜
RETRIEVAL_TOP_K = 15
#
'''
请根据以下【参考资料】回答问题。回答时请尽量引用原文关键句，并用自己的话解释。如果参考资料中没有直接答案，请根据已有信息推测，并注明“推测”。若完全无相关信息，再回答“无法回答”。


回答规则：
1. 只使用参考资料中的信息，不要编造资料中没有的内容
2. 如果参考资料中没有相关信息，根据已有资料进行推测，并需要标明推测的过程和原因。
3. 回答要条理清晰，重点突出
4. 可以适当引用资料中的原文表述
'''