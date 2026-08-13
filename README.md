# 基于 RAG 的知识库智能问答系统

> 创新应用综合实训课程项目
> 技术栈：LangChain + Chroma + DeepSeek + text2vec 中文嵌入模型

## 项目简介

本项目实现了一个基于检索增强生成（RAG, Retrieval-Augmented Generation）技术的本地知识库智能问答系统。用户可以将 PDF、Word、TXT、Markdown 等格式的文档放入知识库，系统会自动对文档进行分段、向量化并存储到 Chroma 向量数据库中。当用户提问时，系统先从知识库中检索最相关的文本片段，再结合大语言模型生成基于原文的精准回答，有效解决了大模型的知识截止和幻觉问题。

## 项目结构

```
rag_project/
├── config.py              # 全局配置文件（API Key、参数等）
├── build_database.py      # 构建向量知识库脚本
├── chat.py                # 命令行版问答机器人
├── app.py                 # 网页版可视化界面（Streamlit）
├── requirements.txt       # 依赖包清单
├── README.md              # 项目说明文档
├── knowledge_base/        # 存放知识库文档（pdf/txt/docx/md）
└── vector_db/             # 向量数据库（运行后自动生成）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置 API Key

打开 `config.py`，将 `DEEPSEEK_API_KEY` 替换为你自己的 DeepSeek API Key。

申请地址：https://platform.deepseek.com/

### 3. 准备知识库文档

将你的文档（pdf/txt/docx/md）放入 `knowledge_base/` 文件夹。

### 4. 构建向量知识库

```bash
python build_database.py
```

### 5. 运行问答系统

**命令行版：**
```bash
python chat.py
```

**网页版（推荐演示用）：**
```bash
streamlit run app.py
```

## 核心技术

| 模块 | 技术选型 | 说明 |
|------|----------|------|
| 开发框架 | LangChain | 大模型应用开发框架，提供 Chains/Retrieval 等组件 |
| 大语言模型 | DeepSeek (deepseek-chat) | 通过 OpenAI 兼容接口调用 |
| 嵌入模型 | text2vec-base-chinese | 中文开源语义向量模型，本地运行 |
| 向量数据库 | Chroma | 轻量级本地向量库，支持持久化存储 |
| 文本分段 | RecursiveCharacterTextSplitter | 递归字符分块，中文优化分隔符 |
| 检索策略 | 余弦相似度语义检索 | 返回 Top-K 最相关文本片段 |
| 网页界面 | Streamlit | 快速构建可视化问答界面 |

## RAG 工作流程

### 离线建库阶段
1. 文档加载：读取 knowledge_base 中的所有文档
2. 文本分段：按语义边界将长文档切分为 500 字符的块
3. 文本向量化：用 text2vec 模型将每个文本块转为 768 维向量
4. 向量存储：将向量和原文存入 Chroma 数据库

### 在线问答阶段
1. 用户输入问题
2. 问题向量化：用同一个嵌入模型将问题转为向量
3. 语义检索：在 Chroma 中计算余弦相似度，返回 Top-3 相关片段
4. Prompt 拼接：将检索片段和用户问题组合成 Prompt
5. 大模型生成：调用 DeepSeek 生成基于原文的回答
6. 返回结果：同时返回答案和引用的原文片段

## 注意事项

- 提交作业时请删除代码中的 API Key
- 向量数据库文件夹 vector_db/ 无需提交，可重新构建
- 知识库原始文档无需全部提交，在报告中说明来源即可
- PDF 必须是可复制文字的版本，扫描版图片 PDF 无法读取
