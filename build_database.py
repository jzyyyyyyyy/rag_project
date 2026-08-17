
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import os
import json
import shutil
from itertools import chain
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from semantic_splitter import split_documents_semantic

from config import (
    KNOWLEDGE_DIR,
    USER_DOCS_DIR,
    VECTOR_DB_PATH,
    SEMANTIC_BREAKPOINT_PERCENTILE,
    SEMANTIC_THRESHOLD_FLOOR,
    SEMANTIC_MAX_CHUNK,
    SEMANTIC_OVERLAP_SENTS,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_NORMALIZE,
)

def load_documents(only_files=None):
    """
    步骤1：加载知识库文件夹中的所有文档
    遍历 knowledge_base（项目自带）与 user_docs（用户上传）两个目录，
    根据后缀选择对应的加载器，支持 .pdf / .txt / .docx / .md
    only_files：只加载指定文件名（用于网页上传后的增量入库）
    """
    print("=" * 60)
    print("步骤1：正在加载知识库文档...")
    print(f" 知识库路径：{KNOWLEDGE_DIR}")
    print(f" 用户上传路径：{USER_DOCS_DIR}")

    docs = []
    # 遍历两个目录
    for root, dirs, files in chain(os.walk(KNOWLEDGE_DIR), os.walk(USER_DOCS_DIR)):
        for file in files:
            if only_files is not None and file not in only_files:
                continue
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

def split_documents(docs, embeddings):
    """
    步骤2：语义分块

    不按固定字符数切分，而是把文本切成句子后，计算相邻句子的语义相似度，
    相似度低于全篇 P15 分位（自适应阈值）处视为话题转换点，在此断开。
    相邻块之间重叠 1 句，缓解边界信息丢失（见 semantic_splitter.py）。
    """
    print("=" * 60)
    print("步骤2：正在进行语义分块...")
    print(f"   断点分位：P{SEMANTIC_BREAKPOINT_PERCENTILE}，下限：{SEMANTIC_THRESHOLD_FLOOR}，"
          f"最大块长：{SEMANTIC_MAX_CHUNK}，重叠：{SEMANTIC_OVERLAP_SENTS} 句")

    split_docs = split_documents_semantic(
        docs,
        embeddings,
        percentile=SEMANTIC_BREAKPOINT_PERCENTILE,
        floor=SEMANTIC_THRESHOLD_FLOOR,
        max_chunk_size=SEMANTIC_MAX_CHUNK,
        overlap_sents=SEMANTIC_OVERLAP_SENTS,
    )
    print(f"切分完成，共得到 {len(split_docs)} 个语义块")
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

def save_chunks_json(split_docs):
    """
    把切分好的文本块保存为 JSON 文件，
    供运行时构建 BM25 关键词检索器复用（避免每次启动重复切分）
    """
    chunks_file = os.path.join(VECTOR_DB_PATH, "chunks.json")
    data = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in split_docs
    ]
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"文本块已保存到：{chunks_file}")


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
        print("错误：知识库文件夹中没有找到文档！")
        print(" 请把 pdf/txt/docx/md 文件放进 knowledge_base 或 user_docs 文件夹后再运行。")
        return

    # 2. 初始化嵌入模型（语义分块和向量化共用）
    embeddings = build_embeddings()

    # 3. 语义分块
    split_docs = split_documents(docs, embeddings)

    # 4. 保存到向量库（内部会先清空旧库）
    db = save_to_chroma(split_docs, embeddings)

    # 5. 保存文本块（供 BM25 关键词检索复用，须在清库之后写入）
    save_chunks_json(split_docs)

    print("\n" + "=" * 60)
    print("知识库构建全部完成！")
    print(" 接下来可以运行 chat.py 进行问答测试")
    print("=" * 60)


if __name__ == "__main__":
    main()
