[![Tests](https://github.com/Zeng0713/modular-rag-knowledge-base/actions/workflows/tests.yml/badge.svg)](https://github.com/Zeng0713/modular-rag-knowledge-base/actions/workflows/tests.yml)

# Modular RAG Knowledge Base

一个基于 **LangChain、Chroma、BGE-M3 和 DeepSeek** 的模块化知识库问答项目，提供传统 RAG 问答和 Agentic RAG 两种命令行入口。

项目从 Markdown 文档加载并按标题切分知识，使用 SiliconFlow 提供的 BGE-M3 Embedding，存入本地 Chroma 向量数据库。检索结合标题关键词匹配与向量搜索；传统 RAG 流程将检索上下文交给 DeepSeek 并生成答案及引用。Agent 流程由 DeepSeek 根据问题选择工具，执行后再根据工具结果生成回答。

## 功能

- 从 `data/documents/` 加载 Markdown 文档并按标题切分
- 通过 SiliconFlow API 使用 `BAAI/bge-m3` 生成向量
- 使用 Chroma 持久化知识库，并增量新增或更新文档
- 结合标题关键词匹配和向量搜索检索相关内容
- 使用 DeepSeek 生成基于检索资料的回答
- 提供知识库 Agent 工具调用，可搜索知识或列出资料来源
- 传统 RAG 回答支持来源引用
- 集中管理配置，并在启动时检查 API Key
- 使用 Pytest 测试模块；默认跳过需要外部服务的集成测试

## 架构

### 传统 RAG：`rag_qa.py`

```text
data/documents/*.md
        ↓
Document Loader：读取并按标题切分
        ↓
BGE-M3 Embedding（SiliconFlow）
        ↓
Chroma：本地持久化与文档同步
        ↓
Hybrid Retriever：标题关键词匹配 + 向量搜索
        ↓
检索资料 → DeepSeek（生成答案）→ 答案与引用来源
```

### Agentic RAG：`agent_qa.py`

```text
用户问题 → DeepSeek 工具选择（TOOL_SCHEMAS）
                         ↓
          Tool Registry（名称映射与查找）
                         ↓
          Tool Executor（参数分发与执行）
             ↙                         ↘
knowledge_search              list_knowledge_sources
             ↘                         ↙
         工具结果加入对话 → DeepSeek 生成最终回答
```

Agent 当前提供两个工具：`knowledge_search` 使用混合检索查找知识库内容并返回内容及来源；`list_knowledge_sources` 返回已加载文档的来源列表。`src/agent.py` 负责将工具 Schema 提供给模型、执行模型请求、调用 Executor，并将工具结果交回模型生成最终回答；如果模型未选择工具，则直接返回模型回复。

## 项目结构

```text
.
├── data/documents/              # Markdown 知识库
├── src/
│   ├── agent.py                 # Agent 工具选择与调用循环
│   ├── config.py                # 配置与启动校验
│   ├── document_loader.py       # Markdown 加载与切分
│   ├── embeddings.py            # BGE-M3 Embedding
│   ├── exceptions.py            # 自定义服务异常
│   ├── logger.py                # 日志配置
│   ├── rag_chain.py             # DeepSeek 与答案、引用生成
│   ├── retriever.py              # 混合检索
│   ├── tool_registry.py         # 工具注册与查找
│   ├── tool_schemas.py          # 发给模型的工具 Schema
│   ├── tools.py                 # 知识搜索与来源列表工具
│   └── vector_store.py          # Chroma 创建与文档同步
├── tests/                       # 单元测试与集成测试
├── agent_qa.py                  # Agentic RAG 命令行入口
├── rag_qa.py                    # 传统 RAG 命令行入口
├── requirements.txt
├── pytest.ini
└── .env.example                 # 环境变量名称示例
```

## 环境准备

需要 **Python 3.12**。从项目目录执行：

```bash
git clone https://github.com/Zeng0713/modular-rag-knowledge-base.git
cd modular-rag-knowledge-base
python -m venv .venv
```

激活虚拟环境：

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 配置

将 `.env.example` 复制为项目根目录下的 `.env`，并填入你自己的 API Key：

```dotenv
SILICONFLOW_API_KEY=your_siliconflow_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

- `SILICONFLOW_API_KEY`：调用 SiliconFlow 上的 BGE-M3 Embedding 服务。
- `DEEPSEEK_API_KEY`：调用 DeepSeek 对话模型及工具调用。

项目当前将模型名、服务地址和 Chroma 路径设为 `src/config.py` 中的固定值；只有以上两个 API Key 从环境变量读取。`.env` 已加入 `.gitignore`，请勿提交实际密钥。

## 运行

请先将 Markdown 知识文件放入 `data/documents/`。两种入口启动时都会检查配置、加载文档、创建 Embedding 和 Chroma，并同步知识库；首次运行需要访问对应外部 API。

传统 RAG 问答：

```bash
python rag_qa.py
```

Agentic RAG 问答：

```bash
python agent_qa.py
```

按程序提示输入问题。输入 `exit` 或 `退出` 可以结束 Agent；空问题会提示重新输入。知识库为空时，Agent 会提示先添加 Markdown 文档；Embedding 或 DeepSeek 服务临时不可用时会提示重试，不会直接结束交互程序。

## 测试与覆盖率

默认测试是离线测试，不运行标记为 `integration` 的外部服务测试：

```bash
pytest
```

本地查看源码覆盖率：

```bash
pytest --cov=src --cov-report=term-missing
```

只运行集成测试（需要有效的 SiliconFlow / DeepSeek API Key、网络连接及相应服务可用）：

```bash
pytest -o addopts= -m integration
```

当前仓库验证结果：默认测试 **40 passed, 4 deselected**，`src/` 总覆盖率 **96%（220 statements，9 missed）**。最近验证日期：2026-10-30。集成测试未在此次验证中运行。

## CI

GitHub Actions 工作流位于 `.github/workflows/tests.yml`，在推送到 `main`、针对 `main` 的 Pull Request 以及手动触发时运行。工作流使用 Python 3.12，安装 `requirements.txt` 中的依赖，执行带覆盖率报告的默认（非集成）测试，并上传 `coverage.xml` Artifact。

上方徽章链接到 [`Zeng0713/modular-rag-knowledge-base` 的 Tests 工作流](https://github.com/Zeng0713/modular-rag-knowledge-base/actions/workflows/tests.yml)，可在那里查看最新运行状态；本 README 不预设 CI 当前是否通过。
