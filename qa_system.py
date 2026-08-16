# -*- coding: utf-8 -*-
"""
问答系统核心组件（app.py 与 chat.py 共用，避免重复代码）：
1. 嵌入模型 / LLM 初始化
2. 混合检索器：Chroma 向量检索 + BM25 关键词检索，按权重融合
3. 多轮对话问题改写链：结合历史把问题改写成独立查询，解决代词指代
4. 问答链 + 统一提问入口（含检索结果相似度排序）
"""

import json
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from config import (
    VECTOR_DB_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_NORMALIZE,
    RETRIEVAL_TOP_K,
    HYBRID_WEIGHTS,
    DEEPSEEK_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    PROMPT_TEMPLATES,
    DEFAULT_PROMPT_STYLE,
    CONDENSE_PROMPT,
)


def build_embeddings():
    """初始化本地中文嵌入模型（text2vec）"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE},
    )


def build_llm():
    """初始化 DeepSeek 大模型（OpenAI 兼容接口）"""
    return ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )


def load_chunks():
    """从建库时保存的 chunks.json 恢复文本块，用于构建 BM25 检索器"""
    chunks_file = os.path.join(VECTOR_DB_PATH, "chunks.json")
    with open(chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Document(page_content=item["content"], metadata=item["metadata"])
        for item in data
    ]


def build_hybrid_retriever(embeddings):
    """
    混合检索器：
    - 向量检索：语义相似（能理解同义表达）
    - BM25 检索：关键词命中（能精确匹配专有名词）
    用 EnsembleRetriever（RRF 倒数排名融合）按权重合并两条通道的结果。
    返回 (ensemble_retriever, chroma_db)
    """
    db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    vector_retriever = db.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
    bm25_retriever = BM25Retriever.from_documents(load_chunks(), k=RETRIEVAL_TOP_K)
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=HYBRID_WEIGHTS,
    )
    return ensemble, db


def build_condense_chain(llm):
    """
    多轮对话问题改写链：
    结合历史对话，把当前问题改写成完整独立的检索查询。
    例如："它的核心思想是什么？" -> "张岱年《中国文化精神》的核心思想是什么？"
    """
    prompt = ChatPromptTemplate.from_template(CONDENSE_PROMPT)
    return prompt | llm | StrOutputParser()


def build_qa_chain(llm, prompt_style=DEFAULT_PROMPT_STYLE):
    """
    问答链：参考资料 + 用户问题 -> 回答。
    prompt_style 为 PROMPT_TEMPLATES 中的键（严谨/平衡/宽松）
    """
    template = PROMPT_TEMPLATES.get(prompt_style, PROMPT_TEMPLATES[DEFAULT_PROMPT_STYLE])
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()


def format_chat_history(history):
    """history 为 [(角色, 内容), ...] 列表 -> 拼接成文本"""
    if not history:
        return "无历史对话"
    return "\n".join(f"{role}: {content}" for role, content in history)


def format_docs(docs):
    """检索到的文档片段 -> 拼接为上下文字符串"""
    return "\n\n".join(doc.page_content for doc in docs)


def ask(qa_chain, condense_chain, retriever, db, question, chat_history):
    """
    完整问答流程：
    1. 问题改写（结合历史对话，解决代词指代）
    2. 混合检索（向量 + BM25 融合）
    3. 检索结果按向量距离升序排列（即相似度从高到低），供界面展示
    4. 拼接上下文生成回答
    返回 (answer, sorted_docs)，sorted_docs 为 [(doc, 距离分数)]，分数越小越相似
    """
    history_text = format_chat_history(chat_history)
    standalone_q = condense_chain.invoke(
        {"question": question, "chat_history": history_text}
    )
    standalone_q = standalone_q.strip()

    docs = retriever.invoke(standalone_q)

    # 用向量库给每个检索结果打分，用于相似度排序展示
    scored = db.similarity_search_with_score(standalone_q, k=RETRIEVAL_TOP_K)
    score_map = {}
    for doc, score in scored:
        if doc.page_content not in score_map:
            score_map[doc.page_content] = score
    sorted_docs = sorted(
        [(doc, score_map.get(doc.page_content, float("inf"))) for doc in docs],
        key=lambda x: x[1],
    )

    answer = qa_chain.invoke(
        {"question": question, "context": format_docs(docs)}
    )
    return answer, sorted_docs
