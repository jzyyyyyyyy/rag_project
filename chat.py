# -*- coding: utf-8 -*-
"""
命令行版问答机器人
特性：语义分块向量库 + 混合检索（向量+BM25） + 多轮对话记忆（问题改写）
"""

from qa_system import (
    build_embeddings,
    build_llm,
    build_hybrid_retriever,
    build_condense_chain,
    build_qa_chain,
    ask,
)
from config import DEFAULT_PROMPT_STYLE


def main():
    print("=" * 60)
    print(" RAG 本地知识库问答助手 ")
    print(" 输入问题开始提问，输入 exit / 退出 结束程序")
    print("=" * 60)

    print(" 正在初始化系统（加载嵌入模型 + 构建混合检索器）...")
    embeddings = build_embeddings()
    llm = build_llm()
    retriever, db = build_hybrid_retriever(embeddings)
    condense_chain = build_condense_chain(llm)
    qa_chain = build_qa_chain(llm, DEFAULT_PROMPT_STYLE)
    print("系统初始化完成")

    history = []  # [(角色, 内容), ...] 多轮对话记忆

    while True:
        query = input("\n 请输入你的问题：").strip()

        if query.lower() in ["exit", "quit", "退出", "q"]:
            print("\n 感谢使用，再见！")
            break

        if not query:
            continue

        print("正在检索资料并生成回答...")

        try:
            answer, sorted_docs = ask(
                qa_chain, condense_chain, retriever, db, query, history
            )
        except Exception as e:
            print(f"发生错误：{e}")
            continue

        print("\n" + "-" * 60)
        print("回答：")
        print(answer)
        print("-" * 60)

        # 展示检索到的参考片段（按相似度从高到低排序）
        print("\n" + "=" * 60)
        print(f"参考片段（共 {len(sorted_docs)} 个，按相似度从高到低，显示前 5 个）：")
        for i, (doc, score) in enumerate(sorted_docs[:5]):
            snippet = doc.page_content.replace("\n", " ").strip()[:150]
            score_text = "未知" if score == float("inf") else f"{score:.4f}"
            print(f"  [{i + 1}] 距离分数 {score_text}（越小越相似）: {snippet}...")
        print("=" * 60)

        history.append(("用户", query))
        history.append(("助手", answer))
        history = history[-12:]  # 只保留最近 6 轮


if __name__ == "__main__":
    main()
