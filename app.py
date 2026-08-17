import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from config import (
    KNOWLEDGE_DIR,
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

# ---------- 页面基础配置 ----------
st.set_page_config(
    page_title="RAG 知识库智能问答助手",
    page_icon="📚",
    layout="wide",
)

st.title("📚 基于 RAG 的知识库智能问答助手")
st.caption(
    "技术栈：LangChain + Chroma 向量数据库 + DeepSeek 大模型 + text2vec 中文嵌入模型"
)
st.divider()


# ---------- 缓存加载系统 ----------
@st.cache_resource
def init_qa_system():
    """初始化整个问答系统，结果缓存，只运行一次"""
    # 1. 嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE},
    )

    # 2. 加载 Chroma 向量库
    db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
    )
    retriever = db.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})

    # 3. 大语言模型
    llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    # 4. Prompt 模板
    prompt_template = """
请根据以下【参考资料】回答问题。回答时请尽量引用原文关键句，并用自己的话解释。如果参考资料中没有直接答案，请根据已有信息推测，并注明"推测"。若完全无相关信息，再回答"无法回答"。

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
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 5. 构建 LCEL 链
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
            RunnableParallel(
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
    )

    return qa_chain, retriever, embeddings, db


# ---------- 初始化系统 ----------
with st.spinner("系统正在初始化，请稍候（第一次加载嵌入模型需要几分钟）..."):
    qa_chain, retriever, embeddings, db = init_qa_system()

st.success("✅ 系统初始化完成，可以开始提问了！")
st.divider()

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("⚙️ 系统信息")
    st.info(
        f"""
        **嵌入模型**：{EMBEDDING_MODEL_NAME}

        **大语言模型**：{LLM_MODEL}

        **向量数据库**：Chroma（本地）

        **检索片段数**：Top-{RETRIEVAL_TOP_K}

        **分块大小**：256 字符
        """
    )
    st.divider()
    
    st.header("📖 使用说明")
    st.write("1. 在下方输入框输入你的问题")
    st.write("2. 按回车发送，系统会自动检索知识库")
    st.write("3. 回答下方可展开查看参考原文")
    st.write("4. 问题答案都来自你上传的文档")
    
    st.divider()

    st.header("📚 知识库文档")
    kb_files = sorted(
        f for f in os.listdir(KNOWLEDGE_DIR)
        if os.path.isfile(os.path.join(KNOWLEDGE_DIR, f))
    )
    if kb_files:
        for name in kb_files:
            st.write(f"📄 {name}")
    else:
        st.write("（暂无文档）")

    st.divider()
    st.header("📤 上传文档到知识库")
    if "upload_round" not in st.session_state:
        st.session_state.upload_round = 0
    uploaded_files = st.file_uploader(
        "支持 PDF / TXT / DOCX / MD",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.upload_round}"
    )
    if uploaded_files:
        if st.button("🚀 添加到知识库"):
            with st.spinner("正在保存并向量化新文档，请稍候..."):
                import json
                from config import (
                    KNOWLEDGE_DIR,
                    SEMANTIC_BREAKPOINT_PERCENTILE,
                    SEMANTIC_THRESHOLD_FLOOR,
                    SEMANTIC_MAX_CHUNK,
                    SEMANTIC_OVERLAP_SENTS,
                )
                from build_database import load_documents
                from semantic_splitter import split_documents_semantic

                # 1. 保存文件到 knowledge_base（重名直接覆盖）
                new_files = []
                for uploaded_file in uploaded_files:
                    save_path = os.path.join(KNOWLEDGE_DIR, uploaded_file.name)
                    if os.path.exists(save_path):
                        st.info(f"📝 {uploaded_file.name} 已存在，将覆盖旧版本")
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    new_files.append(uploaded_file.name)

                # 2. 只加载新文件并语义分块
                docs = load_documents(only_files=new_files)
                split_docs = split_documents_semantic(
                    docs,
                    embeddings,
                    percentile=SEMANTIC_BREAKPOINT_PERCENTILE,
                    floor=SEMANTIC_THRESHOLD_FLOOR,
                    max_chunk_size=SEMANTIC_MAX_CHUNK,
                    overlap_sents=SEMANTIC_OVERLAP_SENTS,
                )

                # 3. 先删除这些文件旧的向量（重名覆盖时保证库里只有新内容）
                chunks_file = os.path.join(VECTOR_DB_PATH, "chunks.json")
                old_chunks = []
                if os.path.exists(chunks_file):
                    with open(chunks_file, "r", encoding="utf-8") as f:
                        old_chunks = json.load(f)
                for name in new_files:
                    source = os.path.join(KNOWLEDGE_DIR, name)
                    db.delete(where={"source": source})
                    old_chunks = [
                        c for c in old_chunks
                        if c["metadata"].get("source") != source
                    ]

                # 4. 增量写入向量库（不清空其他文档的数据）
                db.add_documents(split_docs)

                # 5. 更新 chunks.json（供 BM25 检索复用）
                new_chunks = old_chunks + [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in split_docs
                ]
                with open(chunks_file, "w", encoding="utf-8") as f:
                    json.dump(new_chunks, f, ensure_ascii=False, indent=2)

                st.success(f"✅ 已添加 {len(new_files)} 个文档：{', '.join(new_files)}")
                st.info(f"💡 共向量化 {len(split_docs)} 个文本块，现在可以直接提问了")

                # 重置上传组件，清空已选文件
                st.session_state.upload_round += 1

# ---------- 聊天界面 ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("请输入你想问的问题..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索知识库并生成回答..."):
            source_docs = retriever.invoke(user_input)
            answer = qa_chain.invoke(user_input)

        st.markdown(answer)

        # ===== 用户反馈按钮 =====
        feedback_key = f"feedback_{len(st.session_state.chat_history)}"
        
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("👍 有用", key=f"{feedback_key}_like"):
                st.session_state[feedback_key] = "liked"
                st.success("感谢您的反馈！")
        with col2:
            if st.button("👎 无用", key=f"{feedback_key}_dislike"):
                st.session_state[feedback_key] = "disliked"
                st.warning("请告诉我们哪里有问题：")
                feedback_text = st.text_area("反馈内容", key=f"{feedback_key}_text")
                if st.button("提交反馈", key=f"{feedback_key}_submit"):
                    with open("feedback_log.txt", "a", encoding="utf-8") as f:
                        f.write(f"问题: {user_input}\n")
                        f.write(f"回答: {answer}\n")
                        f.write(f"反馈: {feedback_text}\n")
                        f.write("-" * 50 + "\n")
                    st.success("感谢您的反馈，已记录！")

        with st.expander("📖 查看参考原文片段"):
            for i, doc in enumerate(source_docs):
                source = doc.metadata.get("source", "未知")
                st.markdown(f"**片段 {i + 1}**（来源：{source}）")
                st.write(doc.page_content)
                st.divider()

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )