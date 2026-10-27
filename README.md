# Agentic RAG Knowledge Base

一个基于 LangChain、Chroma、SiliconFlow Embedding API 和 DeepSeek 构建的模块化 RAG（Retrieval-Augmented Generation）知识库问答项目。

项目实现了从 Markdown 文档加载、知识切分、向量化、Chroma 持久化存储、混合检索，到基于 DeepSeek 的答案生成与来源引用的完整 RAG 流程。

## 项目特点

- 支持 Markdown 知识库加载与按标题切分
- 使用 BGE-M3 生成文本向量
- 使用 Chroma 实现向量持久化存储
- 实现标题关键词匹配 + 向量检索的混合检索策略
- 使用 DeepSeek 根据检索结果生成回答
- 支持回答来源引用（Citation）
- 支持知识库增量同步，避免重复向量化
- 使用统一配置模块管理模型、API 和数据库配置
- 使用日志系统记录知识库同步和 API 异常
- 对 Embedding 和 LLM 外部服务实现独立异常处理
- 提供模块化测试，覆盖配置、检索、向量数据库和 RAG Chain 等核心模块

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
git clone <your-repository-url>
cd <your-repository-name>
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

