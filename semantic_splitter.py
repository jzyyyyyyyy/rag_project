# -*- coding: utf-8 -*-
"""
语义分块器：不按固定字符数切分，而是按"句子间语义相似度"寻找自然断点。
思路：相邻句子语义相近 -> 属于同一话题，继续合并；
      相似度骤降 -> 话题转换，在此断开。
相比固定字符切分，不会把完整语义从句子中间拦腰截断。

泛用性设计：
1. 断点阈值按全篇相邻句相似度的百分位自适应确定（默认取 P15），
   换任何书籍/语料无需手工校准；另设绝对下限，防止相似度整体偏高时不断开。
2. 相邻块之间重叠 1 句，缓解话题边界处信息丢失。
"""

import re

import numpy as np
from langchain_core.documents import Document

# 按中文标点切句，保留标点本身
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；])\s*")


def _clean_text(text):
    """
    清洗 PDF 抽取出的文本：
    - PDF 每行结尾都有换行符（排版换行，不是段落边界），统一替换为空白
    - 合并多余空白字符
    若保留换行切句，每个排版行都会被当成一句话，导致语义块碎成几个字。
    """
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def split_sentences(text):
    """把一段文本切成句子列表（保留标点）"""
    return [p.strip() for p in _SENT_SPLIT_RE.split(text) if p and p.strip()]


def _cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _embed_sentences(docs, embeddings):
    """
    把每篇文档切成句子并一次性向量化。
    返回 [(sentences, vectors), ...]；单句文档 vectors 为 None。
    句子向量只计算这一次，同时用于阈值统计和分块，不重复计算。
    """
    result = []
    for doc in docs:
        sents = split_sentences(_clean_text(doc.page_content))
        if not sents:
            result.append(([], None))
            continue
        if len(sents) == 1:
            result.append((sents, None))
            continue
        result.append((sents, embeddings.embed_documents(sents)))
    return result


def _compute_threshold(sent_vec_pairs, percentile, floor):
    """
    按全篇（所有文档）相邻句相似度分布计算断点阈值：
    threshold = max(floor, 相似度第 percentile 百分位)
    """
    sims = []
    for sents, vecs in sent_vec_pairs:
        if vecs is None:
            continue
        for i in range(1, len(sents)):
            sims.append(_cosine(vecs[i], vecs[i - 1]))
    if not sims:
        return floor
    return max(floor, float(np.percentile(sims, percentile)))


def _chunk_sentences(sents, vecs, threshold, max_chunk_size, overlap_sents):
    """
    按阈值把句子合并成块：
    - 相邻句相似度低于阈值 -> 话题转换，断开
    - 块长超过上限 -> 断开
    - 断开后新块带上上一块末尾 overlap_sents 句作为重叠
    """
    if not sents:
        return []
    if vecs is None:
        return ["".join(sents)]

    chunks = []
    cur = [sents[0]]
    cur_len = len(sents[0])
    for i in range(1, len(sents)):
        too_dissimilar = _cosine(vecs[i], vecs[i - 1]) < threshold
        too_long = cur_len + len(sents[i]) > max_chunk_size
        if too_dissimilar or too_long:
            chunks.append("".join(cur))
            cur = cur[-overlap_sents:] + [sents[i]]
            cur_len = sum(len(s) for s in cur)
        else:
            cur.append(sents[i])
            cur_len += len(sents[i])
    if cur:
        chunks.append("".join(cur))
    return chunks


def split_documents_semantic(
    docs,
    embeddings,
    percentile=15,
    floor=0.35,
    max_chunk_size=500,
    overlap_sents=1,
):
    """
    对一批 Document 做语义分块，返回新的 Document 列表。
    - percentile: 断点阈值取全篇相邻句相似度分布的该百分位（自适应，换语料不用改）
    - floor: 阈值绝对下限，低于它必定断开
    - max_chunk_size: 单个块字符数上限
    - overlap_sents: 相邻块重叠句数
    每个块继承原文档的 metadata（source 等）。
    """
    sent_vec_pairs = _embed_sentences(docs, embeddings)
    threshold = _compute_threshold(sent_vec_pairs, percentile, floor)

    new_docs = []
    for doc, (sents, vecs) in zip(docs, sent_vec_pairs):
        for chunk_text in _chunk_sentences(
            sents, vecs, threshold, max_chunk_size, overlap_sents
        ):
            new_docs.append(
                Document(page_content=chunk_text, metadata=dict(doc.metadata))
            )
    return new_docs
