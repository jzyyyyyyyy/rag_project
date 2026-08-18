import os
import json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st

from qa_system import (
    build_embeddings,
    build_llm,
    build_hybrid_retriever,
    build_condense_chain,
    build_qa_chain,
    ask,
)
from config import (
    KNOWLEDGE_DIR,
    USER_DOCS_DIR,
    VECTOR_DB_PATH,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL,
    RETRIEVAL_TOP_K,
    PROMPT_TEMPLATES,
    DEFAULT_PROMPT_STYLE,
)

# 确保目录存在（首次运行或目录被删除时自动创建，避免报错）
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(USER_DOCS_DIR, exist_ok=True)

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
    """初始化系统资源（嵌入模型 / LLM / 混合检索器 / 向量库），结果缓存，只运行一次"""
    embeddings = build_embeddings()
    llm = build_llm()
    retriever, db = build_hybrid_retriever(embeddings)
    return llm, retriever, db, embeddings


def cleanup_orphan_chunks(db, chunks_file):
    """
    删除向量库中源文件已不存在的文本块（如手动删除了 user_docs 里的文件）。
    同步清理 Chroma 与 chunks.json，返回清理的条数。
    """
    valid_sources = set()
    for base in (KNOWLEDGE_DIR, USER_DOCS_DIR):
        if os.path.isdir(base):
            valid_sources.update(
                os.path.join(base, f)
                for f in os.listdir(base)
                if os.path.isfile(os.path.join(base, f))
            )

    metadatas = db._collection.get(include=["metadatas"])["metadatas"]
    orphan_sources = {
        m.get("source")
        for m in metadatas
        if m and m.get("source") not in valid_sources
    }
    cleaned = 0
    for source in orphan_sources:
        db.delete(where={"source": source})
        cleaned += 1
    if cleaned > 0 and os.path.exists(chunks_file):
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        chunks = [
            c for c in chunks
            if c["metadata"].get("source") not in orphan_sources
        ]
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    return cleaned


# ---------- 初始化系统 ----------
with st.spinner("系统正在初始化，请稍候（第一次加载嵌入模型需要几分钟）..."):
    llm, retriever, db, embeddings = init_qa_system()
    orphan_count = cleanup_orphan_chunks(
        db, os.path.join(VECTOR_DB_PATH, "chunks.json")
    )

if orphan_count:
    st.warning(f"🧹 已自动清理 {orphan_count} 条失效文档的向量（源文件已不存在）")

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

        **分块方式**：语义分块（≤500 字符）
        """
    )
    st.divider()

    st.header("🎛️ 回答风格")
    style_options = list(PROMPT_TEMPLATES.keys())
    default_index = (
        style_options.index(DEFAULT_PROMPT_STYLE)
        if DEFAULT_PROMPT_STYLE in style_options
        else 0
    )
    st.selectbox(
        "选择回答风格",
        style_options,
        index=default_index,
        key="prompt_style",
    )
    st.caption("严谨：禁止推测 / 平衡：推测须标注 / 宽松：灵活回答")

    st.divider()

    st.header("📖 使用说明")
    st.write("1. 在下方输入框输入你的问题")
    st.write("2. 按回车发送，系统会自动检索知识库")
    st.write("3. 回答下方可展开查看参考原文")
    st.write("4. 问题答案都来自你上传的文档")

    st.divider()

    st.header("📚 知识库文档")
    st.caption("项目自带:")
    kb_files = sorted(
        f for f in os.listdir(KNOWLEDGE_DIR)
        if os.path.isfile(os.path.join(KNOWLEDGE_DIR, f))
    )
    if kb_files:
        for name in kb_files:
            st.write(f"📄 {name}")
    else:
        st.write("（无）")
    st.caption("用户上传:")
    user_files = sorted(
        f for f in os.listdir(USER_DOCS_DIR)
        if os.path.isfile(os.path.join(USER_DOCS_DIR, f))
    )
    if user_files:
        for name in user_files:
            st.write(f"📄 {name}")
    else:
        st.write("（无）")

    st.divider()
    st.header("📤 上传文档到知识库")
    if "upload_round" not in st.session_state:
        st.session_state.upload_round = 0
    if "upload_notice" not in st.session_state:
        st.session_state.upload_notice = None
    if st.session_state.upload_notice:
        st.success(st.session_state.upload_notice)
        st.session_state.upload_notice = None
    uploaded_files = st.file_uploader(
        "支持 PDF / TXT / DOCX / MD",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.upload_round}"
    )
    if uploaded_files:
        if st.button("🚀 添加到知识库"):
            with st.spinner("正在保存并向量化新文档，请稍候..."):
                from config import (
                    SEMANTIC_BREAKPOINT_PERCENTILE,
                    SEMANTIC_THRESHOLD_FLOOR,
                    SEMANTIC_MAX_CHUNK,
                    SEMANTIC_OVERLAP_SENTS,
                )
                from build_database import load_documents
                from semantic_splitter import split_documents_semantic

                # 1. 保存文件到 user_docs（重名直接覆盖）
                new_files = []
                overwritten = []
                for uploaded_file in uploaded_files:
                    save_path = os.path.join(USER_DOCS_DIR, uploaded_file.name)
                    if os.path.exists(save_path):
                        overwritten.append(uploaded_file.name)
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
                    source = os.path.join(USER_DOCS_DIR, name)
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

                notice = f"✅ 已添加 {len(new_files)} 个文档：{', '.join(new_files)}"
                if overwritten:
                    notice += f"（{'、'.join(overwritten)} 已覆盖旧版本）"
                notice += f"，共向量化 {len(split_docs)} 个文本块，现在可以直接提问了"
                st.session_state.upload_notice = notice

                # 重置上传组件，清空已选文件
                st.session_state.upload_round += 1
                st.rerun()

    st.divider()
    if st.button("🧹 清理失效文档向量"):
        with st.spinner("正在清理失效向量..."):
            cleaned = cleanup_orphan_chunks(
                db, os.path.join(VECTOR_DB_PATH, "chunks.json")
            )
        if cleaned:
            st.success(f"已清理 {cleaned} 条失效文档向量")
        else:
            st.success("没有发现失效文档向量")

# ---------- 聊天界面 ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] != "assistant":
            continue

        msg_id = msg["id"]

        # ===== 用户反馈按钮 =====
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("👍 有用", key=f"fb_{msg_id}_like"):
                st.session_state[f"fb_{msg_id}_verdict"] = "liked"
        with col2:
            if st.button("👎 无用", key=f"fb_{msg_id}_dislike"):
                st.session_state[f"fb_{msg_id}_verdict"] = "disliked"
                st.session_state[f"fb_{msg_id}_show"] = True

        if st.session_state.get(f"fb_{msg_id}_verdict") == "liked":
            st.success("感谢您的反馈！")

        # ===== 反馈输入面板（点差评后弹出） =====
        if st.session_state.get(f"fb_{msg_id}_show"):
            st.warning("请告诉我们哪里有问题：")
            st.text_area("反馈内容", key=f"fb_{msg_id}_text")
            if st.button("提交反馈", key=f"fb_{msg_id}_submit"):
                with open("feedback_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"问题: {msg.get('question', '')}\n")
                    f.write(f"回答: {msg['content']}\n")
                    f.write(
                        f"反馈: {st.session_state.get(f'fb_{msg_id}_text', '')}\n"
                    )
                    f.write("-" * 50 + "\n")
                st.session_state[f"fb_{msg_id}_show"] = False
                st.success("感谢您的反馈，已记录！")

        # ===== 参考原文片段 =====
        if msg.get("docs"):
            with st.expander("📖 查看参考原文片段"):
                for i, (content, source, score) in enumerate(msg["docs"]):
                    st.markdown(
                        f"**片段 {i + 1}**（来源：{source}，距离分数 {score:.4f}，越小越相似）"
                    )
                    st.write(content)
                    st.divider()

if user_input := st.chat_input("请输入你想问的问题..."):
    # 多轮对话历史（只保留最近 6 轮，供问题改写链还原指代）
    history = [
        (m["role"], m["content"])
        for m in st.session_state.chat_history[-12:]
    ]

    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    style = st.session_state.get("prompt_style", DEFAULT_PROMPT_STYLE)
    qa_chain = build_qa_chain(llm, style)
    condense_chain = build_condense_chain(llm)

    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索知识库并生成回答..."):
            try:
                answer, sorted_docs = ask(
                    qa_chain, condense_chain, retriever, db, user_input, history
                )
            except Exception as e:
                st.error(f"生成回答失败：{e}")
                st.rerun()
        st.markdown(answer)

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "id": len(st.session_state.chat_history) + 1,
            "question": user_input,
            "docs": [
                (
                    doc.page_content,
                    doc.metadata.get("source", "未知"),
                    float(score),
                )
                for doc, score in sorted_docs
            ],
        }
    )
    st.rerun()
