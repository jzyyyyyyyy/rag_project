

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import (
    VECTOR_DB_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_NORMALIZE,
    DEEPSEEK_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RETRIEVAL_TOP_K,
)


def load_retriever():
    """
    加载向量数据库并返回检索器
    """
    print(" 正在加载向量数据库...")

    # 使用和建库时相同的嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE},
    )

    db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
    )
    retriever = db.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
    print("向量数据库加载完成")
    return retriever


def build_chain():
    """
    构建 RAG 问答链（LCEL 方式），并返回链和检索器
    """
    retriever = load_retriever()

    # 初始化大语言模型（DeepSeek）
    llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    # Prompt 模板
    template = """请根据以下【参考资料】回答问题。回答时请尽量引用原文关键句，并用自己的话解释。如果参考资料中没有直接答案，请根据已有信息推测，并注明“推测”。若完全无相关信息，再回答“无法回答”。
    

回答规则：
1. 只使用参考资料中的信息，不要编造资料中没有的内容
2. 如果参考资料中没有相关信息，根据已有资料进行推测，并需要标明推测的过程和原因。
3. 回答要条理清晰，重点突出
4. 可以适当引用资料中的原文表述

【参考资料】
{context}

【用户问题】
{question}

【你的回答】
"""
    prompt = ChatPromptTemplate.from_template(template)

    # 格式化检索到的文档片段（拼接为字符串）
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 构建 LCEL 链
    chain = (
            {
                "context": retriever | format_docs,  # 检索 -> 拼接
                "question": RunnablePassthrough()  # 用户问题直接传递
            }
            | prompt
            | llm
            | StrOutputParser()
    )
    return chain, retriever   # 返回两个对象


def main():
    print("=" * 60)
    print(" RAG 本地知识库问答助手 ")
    print(" 输入问题开始提问，输入 exit / 退出 结束程序")
    print("=" * 60)

    chain, retriever = build_chain()   # 同时拿到链和检索器

    while True:
        query = input("\n 请输入你的问题：").strip()

        if query.lower() in ["exit", "quit", "退出", "q"]:
            print("\n 感谢使用，再见！")
            break

        if not query:
            continue

        print("正在检索资料并生成回答...")

        # --- 调试：打印检索到的原始片段 ---
        try:
            retrieved_docs = retriever.invoke(query)
            print("\n" + "=" * 60)
            print(f"检索到的片段（共 {len(retrieved_docs)} 个，显示前 5 个）：")
            for i, doc in enumerate(retrieved_docs[:5]):
                # 只显示前 150 个字符，避免刷屏
                snippet = doc.page_content.replace('\n', ' ').strip()[:150]
                print(f"  [{i+1}] {snippet}...")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"调试检索失败：{e}")

        # 正常生成答案
        try:
            answer = chain.invoke(query)
            print("\n" + "-" * 60)
            print("回答：")
            print(answer)
            print("-" * 60)
        except Exception as e:
            print(f"发生错误：{e}")


if __name__ == "__main__":
    main()
