
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import os
import shutil
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    KNOWLEDGE_DIR,
    VECTOR_DB_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_NORMALIZE,
)

def load_documents():
    """
    步骤1：加载知识库文件夹中的所有文档
    手动遍历文件夹，根据后缀选择对应的加载器
    支持 .pdf / .txt / .docx / .md
    """
    print("=" * 60)
    print("步骤1：正在加载知识库文档...")
    print(f" 知识库路径：{KNOWLEDGE_DIR}")

    docs = []
    # 遍历文件夹
    for root, dirs, files in os.walk(KNOWLEDGE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif ext == ".txt":
                    # 指定 utf-8 编码，避免乱码
                    loader = TextLoader(file_path, encoding="utf-8")
                elif ext == ".docx":
                    loader = UnstructuredWordDocumentLoader(file_path)
                elif ext == ".md":
                    loader = TextLoader(file_path, encoding="utf-8")
                else:
                    print(f"   跳过不支持的文件：{file}")
                    continue
                # 加载该文件的所有 Document 对象
                file_docs = loader.load()
                docs.extend(file_docs)
                print(f"   已加载：{file}（{len(file_docs)} 个片段）")
            except Exception as e:
                print(f"   加载失败：{file}，错误：{e}")

    print(f"加载完成，共读取 {len(docs)} 个文档片段")
    return docs

def split_documents(docs):
    """
    步骤2：文本分段

    使用 RecursiveCharacterTextSplitter（多层递归分块），
    并针对中文优化分隔符列表
    """
    print("=" * 60)
    print("步骤2：正在切分文档...")
    print(f"   分块大小：{CHUNK_SIZE}，重叠大小：{CHUNK_OVERLAP}")

    # 课件中给出的中文优化分隔符（实战2第22页）
    # 优先按段落、换行、句号等自然语义边界切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",   # 段落分隔（最高优先级）
            "\n",     # 换行
            "。",     # 中文句号
            "！",     # 中文感叹号
            "？",     # 中文问号
            "，",     # 中文逗号
            "、",     # 中文顿号
            " ",      # 空格
            "",       # 字符级兜底
        ],
    )

    split_docs = text_splitter.split_documents(docs)
    print(f"切分完成，共得到 {len(split_docs)} 个文本块")
    return split_docs


def build_embeddings():
    """
    步骤3：初始化嵌入模型

    使用 HuggingFace 本地中文嵌入模型 text2vec
    嵌入模型是编码器架构（BERT类），输出固定长度语义向量
    """
    print("=" * 60)
    print("步骤3：正在初始化嵌入模型...")
    print(f"   模型：{EMBEDDING_MODEL_NAME}")
    print(f"   设备：{EMBEDDING_DEVICE}")
    print("   （第一次运行会自动下载模型，约400MB，请耐心等待）")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE},
    )
    print("嵌入模型初始化完成")
    return embeddings

def save_to_chroma(split_docs, embeddings):
    """
    步骤4：将向量存入 Chroma 向量数据库（分批添加，避免卡顿）
    """
    print("=" * 60)
    print("步骤4：正在生成向量并保存到 Chroma 数据库...")

    # 如果旧数据库存在，先删除
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
        print("已删除旧的向量数据库")

    # 分批添加，每批 100 个文档，并打印进度
    batch_size = 100
    total = len(split_docs)
    db = None

    for i in range(0, total, batch_size):
        batch = split_docs[i:i+batch_size]
        if db is None:
            # 第一批：创建数据库
            db = Chroma.from_documents(
                batch,
                embeddings,
                persist_directory=VECTOR_DB_PATH,
            )
        else:
            # 后续批次：增量添加
            db.add_documents(batch)
        print(f"已处理 {min(i+batch_size, total)} / {total} 个文本块")

    print(f"向量数据库已保存到：{VECTOR_DB_PATH}")
    return db

def main():
    """主流程：完整的离线建库过程"""
    print("\n" + "=" * 60)
    print("开始构建 RAG 向量知识库")
    print("=" * 60)

    # 1. 加载文档
    docs = load_documents()
    if len(docs) == 0:
        print("错误：knowledge_base 文件夹中没有找到文档！")
        print(" 请把 pdf/txt/docx/md 文件放进 knowledge_base 文件夹后再运行。")
        return

    # 2. 文本分段
    split_docs = split_documents(docs)

    # 3. 初始化嵌入模型
    embeddings = build_embeddings()

    # 4. 保存到向量库
    db = save_to_chroma(split_docs, embeddings)

    print("\n" + "=" * 60)
    print("知识库构建全部完成！")
    print(" 接下来可以运行 chat.py 进行问答测试")
    print("=" * 60)


if __name__ == "__main__":
    main()
