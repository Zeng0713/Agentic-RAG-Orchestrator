# Modular RAG Knowledge Base

一个基于 **LangChain + Chroma + BGE-M3 + DeepSeek** 构建的模块化 RAG（Retrieval-Augmented Generation）知识库问答系统。

项目实现了从 Markdown 文档加载、知识切分、Embedding、Chroma 向量存储、混合检索，到 DeepSeek 生成答案与引用来源的完整 RAG 链路，并通过模块化设计、异常处理和 Pytest 自动化测试提升项目的可维护性与稳定性。

## Features

- Markdown 知识文档自动加载与按标题切分
- 基于 BGE-M3 的文本向量化
- Chroma 向量数据库持久化存储
- 标题关键词匹配 + 向量搜索的 Hybrid Retrieval
- DeepSeek 基于检索上下文生成回答
- 回答引用来源追踪（Citation）
- 知识库增量同步，避免重复写入
- 配置集中管理与启动时 Fail Fast 检查
- Embedding / LLM 服务异常统一处理
- Pytest 单元测试与集成测试隔离

## 技术栈

- Python 3.12
- LangChain
- Chroma
- BGE-M3
- SiliconFlow API
- DeepSeek API
- python-dotenv

## 项目架构

```text
data/documents
      │
      ▼
Document Loader
      │
      ▼
Document Chunks
      │
      ▼
BGE-M3 Embedding
      │
      ▼
Chroma Vector Store
      │
      ▼
Hybrid Retriever
      │
      ├── Title Keyword Match
      │
      └── Vector Search
      │
      ▼
Retrieved Context
      │
      ▼
DeepSeek
      │
      ▼
Answer + Citation

## 项目结构

```text
.
├── data/
│   └── documents/              # Markdown 知识库
│
├── src/
│   ├── config.py               # 项目配置与环境变量校验
│   ├── logger.py               # 日志配置
│   ├── exceptions.py           # 自定义异常
│   ├── document_loader.py      # 文档加载与切分
│   ├── embeddings.py           # Embedding 模型与异常处理
│   ├── vector_store.py         # Chroma 向量数据库
│   ├── retriever.py            # 混合检索
│   └── rag_chain.py            # LLM 生成与 Citation
│
├── tests/                      # 模块测试
├── rag_qa.py                   # CLI 程序入口
├── requirements.txt            # Python 依赖
├── .env.example                # 环境变量示例
├── .gitignore
└── README.md
```

## 安装与运行
### 1. 克隆项目
```bash
git clone https://github.com/Zeng0713/modular-rag-knowledge-base.git
cd modular-rag-knowledge-base
```

### 2. 创建虚拟环境

```bash
python -m venv venv
```

Windows：

```bash
venv\Scripts\activate
```

macOS / Linux：

```bash
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 并重命名为 `.env`：

```text
SILICONFLOW_API_KEY=your_siliconflow_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

请使用自己的 API Key，不要将 `.env` 提交到 Git 仓库。

### 5. 运行项目

```bash
python rag_qa.py
```

运行后即可在命令行中输入问题：

```text
请输入你的问题（输入 exit 退出）：什么是 RAG？
```

输入：

```text
exit
```
即可退出程序。
## Testing

项目使用 Pytest 进行自动化测试，并将普通单元测试与真实外部 API 集成测试分离。

运行默认测试：

```bash
pytest
