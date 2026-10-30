[![Tests](https://github.com/Zeng0713/modular-rag-knowledge-base/actions/workflows/tests.yml/badge.svg)](https://github.com/Zeng0713/modular-rag-knowledge-base/actions/workflows/tests.yml)

# Agentic RAG Orchestrator

*A Modular and Agentic RAG Engine with Hybrid Retrieval, Chroma, and DeepSeek Tool Calling.*

这是一个面向本地知识库问答的模块化项目，包含传统 RAG 流程和基于 DeepSeek 工具调用的 Agentic RAG 入口。Agent 每轮对话可参考最近六轮问答，并最多执行四轮工具调用；达到工具调用上限后会基于已有结果生成最终回答，适合作为 Agentic RAG 实践项目，不定位为通用多 Agent 编排平台。

一个基于 **LangChain、Chroma、BGE-M3 与 DeepSeek** 构建的模块化知识库问答引擎，覆盖 Markdown 文档解析与分块、向量化索引、标题关键词与向量混合检索，以及基于检索上下文的答案生成与来源引用；同时提供传统 RAG 和 Agentic RAG 两种 CLI 流程，支持 DeepSeek 工具选择、Schema 驱动的参数校验与工具结果回传。

项目从 Markdown 文档加载并按标题切分知识，使用 SiliconFlow 提供的 BGE-M3 Embedding，存入本地 Chroma 向量数据库。检索结合标题关键词匹配与向量搜索；传统 RAG 流程将检索上下文交给 DeepSeek 并生成答案及引用。Agent 流程由 DeepSeek 根据问题选择工具，执行后再根据工具结果生成回答。

## 功能

- **结构化知识摄取**：加载 Markdown 文档并按二级标题拆分，保留来源路径与文档类型等元数据。
- **语义向量化**：通过 SiliconFlow 调用 BGE-M3，将知识片段编码为向量表示。
- **持久化知识索引**：使用 Chroma 保存向量与文档内容，支持新增片段及更新已变化内容。
- **混合检索**：融合标题关键词命中与向量相似度搜索，优先保留标题匹配结果并对重复片段去重。
- **基于证据的答案生成**：将检索片段作为上下文交给 DeepSeek，生成回答并整理实际引用的资料来源。
- **Schema 驱动的 Agent 工具调用**：由 DeepSeek 在 `knowledge_search` 与 `list_knowledge_sources` 之间选择，程序通过注册表和执行器完成分发。
- **工具参数校验与结果回传**：执行前依据 Tool Schema 检查参数对象、必填字段、未定义字段和基本类型，再将工具结果回传模型生成答复。
- **有限上下文对话与迭代式工具编排**：在当前会话中保留最近六轮问答，结合对话历史解析指代与连续追问；单个问题最多执行四轮工具调用，并将每轮结果纳入后续决策，达到上限后基于已有证据生成答复。
- **双模式命令行体验**：分别通过 `rag_qa.py` 使用传统 RAG，或通过 `agent_qa.py` 使用 Agentic RAG。
- **配置校验与服务错误处理**：启动时检查所需 API Key，并对 Embedding、LLM 服务错误提供统一异常处理。
- **自动化质量检查**：使用 Pytest 覆盖核心模块；默认测试与真实外部 API 集成测试分开运行。
## 知识库文档

`data/documents/` 中目前包含以下 Markdown 知识文档，程序启动时会读取并按二级标题切分：

- `agent_intro.md`：Agent、RAG、Embedding、向量数据库和 Agent 执行循环基础。
- `rag_intro.md`：RAG 定义、基本流程、优点和局限。
- `rag_bulid_intro.md`：RAG 搭建总览、切分、Embedding、向量存储、检索、回答生成与评估。
- `agentic_rag_build_intro.md`：Agentic RAG 流程、工具 Schema、Registry、Executor、结果回传和评估。

新增或修改文档后，下次运行入口时会将新增或变化的片段同步到本地 Chroma。当前同步逻辑处理新增和更新；删除 Markdown 文件不会自动清理 Chroma 中对应的旧片段。

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
         工具结果加入对话 → 最多四轮工具调用 → DeepSeek 生成最终回答
```

Agent 当前提供两个工具：`knowledge_search` 使用混合检索查找知识库内容并返回内容及来源；`list_knowledge_sources` 返回已加载文档的来源列表。`src/agent.py` 负责将工具 Schema 提供给模型、执行模型请求、调用 Executor，并将工具结果交回模型继续决策，最多循环四轮；达到上限后要求模型基于已有结果直接回答。如果模型未选择工具，则直接返回模型回复。

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

按程序提示输入问题。Agent 会在当前进程中保留最近六轮问答上下文，支持结合前文理解追问；退出程序后历史清空。输入 `exit` 或 `退出` 可以结束 Agent；空问题会提示重新输入。知识库为空时，Agent 会提示先添加 Markdown 文档；Embedding 或 DeepSeek 服务临时不可用时会提示重试，不会直接结束交互程序。

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

当前仓库验证结果：默认测试 **48 passed, 4 deselected**，`src/` 模块覆盖率 **96%（249 statements，10 missed）**；该数字仅统计 `src/`，不包含根目录下的 CLI 入口脚本。最近验证日期：2026-10-30。本轮仅运行默认离线测试；集成测试需单独运行。

## CI

GitHub Actions 工作流位于 `.github/workflows/tests.yml`，在推送到 `main`、针对 `main` 的 Pull Request 以及手动触发时运行。工作流使用 Python 3.12，安装 `requirements.txt` 中的依赖，执行带覆盖率报告的默认（非集成）测试，并上传 `coverage.xml` Artifact。

上方徽章链接到 [`Zeng0713/modular-rag-knowledge-base` 的 Tests 工作流](https://github.com/Zeng0713/modular-rag-knowledge-base/actions/workflows/tests.yml)，可在那里查看最新运行状态；本 README 不预设 CI 当前是否通过。
