import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
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

# ---------- 页面基础配置 ----------
st.set_page_config(
    page_title="RAG 知识库智能问答助手",
    page_icon="",
    layout="wide",
)

st.title("基于 RAG 的知识库智能问答助手")
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
请根据以下【参考资料】回答问题。回答时请尽量引用原文关键句，并用自己的话解释。如果参考资料中没有直接答案，请根据已有信息推测，并注明“推测”。若完全无相关信息，再回答“无法回答”。
    

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

    # 5. 辅助函数
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_chat_history(chat_history):
        if not chat_history:
            return "无历史对话"
        recent = chat_history[-6:]  # 最近3轮
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])

    # 6. 构建 LCEL 链（输入为字典 {"question": ..., "chat_history": [...]}）
    qa_chain = (
        RunnableParallel(
            {
                "context": lambda inputs: format_docs(retriever.invoke(inputs["question"])),
                "question": lambda inputs: f"【历史对话】\n{format_chat_history(inputs['chat_history'])}\n\n【当前问题】\n{inputs['question']}",
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain, retriever


# ---------- 初始化聊天历史（必须在调用 init_qa_system 之前） ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- 初始化系统 ----------
with st.spinner("系统正在初始化，请稍候（第一次加载嵌入模型需要几分钟）..."):
    qa_chain, retriever = init_qa_system()

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

        **检索片段数**：Top-{RETRIEVAL_TOP_K}

        **分块大小**：256 字符
        """
    )
    st.divider()
    st.header("使用说明")
    st.write("1. 在下方输入框输入你的问题")
    st.write("2. 按回车发送，系统会自动检索知识库")
    st.write("3. 回答下方可展开查看参考原文")
    st.write("4. 问题答案都来自你上传的文档")

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
            # 先检索源文档（用于展示引用）
            source_docs = retriever.invoke(user_input)
            # 调用链时传入包含 question 和 chat_history 的字典
            answer = qa_chain.invoke({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })

        st.markdown(answer)

        # 可折叠的参考原文
        with st.expander("查看参考原文片段"):
            for i, doc in enumerate(source_docs):
                source = doc.metadata.get("source", "未知")
                st.markdown(f"**片段 {i + 1}**（来源：{source}）")
                st.write(doc.page_content)
                st.divider()

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )