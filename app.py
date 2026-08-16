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
    EMBEDDING_MODEL_NAME,
    LLM_MODEL,
    RETRIEVAL_TOP_K,
    HYBRID_WEIGHTS,
    PROMPT_TEMPLATES,
    DEFAULT_PROMPT_STYLE,
    SEMANTIC_MAX_CHUNK,
)

# ---------- 页面基础配置 ----------
st.set_page_config(
    page_title="RAG 知识库智能问答助手",
    page_icon="",
    layout="wide",
)

st.title("基于 RAG 的知识库智能问答助手")
st.caption(
    "技术栈：LangChain + Chroma 向量数据库 + DeepSeek 大模型 + text2vec 中文嵌入模型"
    "　|　语义分块 + 混合检索（向量+BM25） + 多轮对话记忆"
)
st.divider()


# ---------- 缓存加载系统 ----------
@st.cache_resource
def init_qa_system():
    """初始化嵌入模型、LLM、混合检索器，结果缓存，只运行一次"""
    embeddings = build_embeddings()
    llm = build_llm()
    retriever, db = build_hybrid_retriever(embeddings)
    condense_chain = build_condense_chain(llm)
    return llm, retriever, db, condense_chain


# ---------- 初始化聊天历史（必须在调用 init_qa_system 之前） ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- 初始化系统 ----------
with st.spinner("系统正在初始化，请稍候（第一次加载嵌入模型需要几分钟）..."):
    llm, retriever, db, condense_chain = init_qa_system()

st.success("系统初始化完成，可以开始提问了！")
st.divider()

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("系统信息")
    st.info(
        f"""
        **嵌入模型**：{EMBEDDING_MODEL_NAME}

        **大语言模型**：{LLM_MODEL}

        **向量数据库**：Chroma（本地）

        **分块方式**：语义分块（≤{SEMANTIC_MAX_CHUNK} 字符）

        **检索方式**：向量 + BM25 混合检索（权重 {HYBRID_WEIGHTS[0]}:{HYBRID_WEIGHTS[1]}）

        **检索片段数**：Top-{RETRIEVAL_TOP_K}
        """
    )
    st.divider()
    st.header("回答风格")
    prompt_style = st.selectbox(
        "选择 Prompt 风格",
        options=list(PROMPT_TEMPLATES.keys()),
        index=list(PROMPT_TEMPLATES.keys()).index(DEFAULT_PROMPT_STYLE),
        help="严谨：资料中没有就直说，禁止推测；平衡：允许推测但必须标明；宽松：回答更灵活",
    )
    st.divider()
    st.header("使用说明")
    st.write("1. 在下方输入框输入你的问题")
    st.write("2. 支持多轮追问（如：它的核心思想是什么？）")
    st.write("3. 回答下方可展开查看参考原文（按相似度从高到低排序）")
    st.write("4. 问题答案都来自你上传的文档")

# ---------- 问答链（Prompt 风格可切换，轻量无需缓存） ----------
qa_chain = build_qa_chain(llm, prompt_style)

# ---------- 聊天界面 ----------
# 显示历史消息
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("请输入你想问的问题..."):
    # 显示用户问题
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并生成回答..."):
            history = [
                (msg["role"], msg["content"])
                for msg in st.session_state.chat_history
            ]
            answer, sorted_docs = ask(
                qa_chain, condense_chain, retriever, db, user_input, history
            )

        st.markdown(answer)

        # 可折叠的参考原文（已按相似度从高到低排序）
        with st.expander("查看参考原文片段（按相似度从高到低排序）"):
            for i, (doc, score) in enumerate(sorted_docs):
                source = doc.metadata.get("source", "未知")
                score_text = "未知" if score == float("inf") else f"{score:.4f}"
                st.markdown(f"**片段 {i + 1}**（来源：{source}｜距离分数：{score_text}，越小越相似）")
                st.write(doc.page_content)
                st.divider()

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )
    # 只保留最近 6 条消息（3 轮对话），避免历史过长
    st.session_state.chat_history = st.session_state.chat_history[-6:]
