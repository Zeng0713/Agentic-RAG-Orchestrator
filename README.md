[![Tests](https://github.com/Zeng0713/Agentic-RAG-Orchestrator/actions/workflows/tests.yml/badge.svg)](https://github.com/Zeng0713/Agentic-RAG-Orchestrator/actions/workflows/tests.yml)

# Agentic RAG Orchestrator

*A Modular and Agentic RAG Engine with Hybrid Retrieval, Chroma, and DeepSeek Tool Calling.*

基于 **LangChain、Chroma、BGE-M3 与 DeepSeek** 的模块化知识库问答项目，提供传统 RAG 与 Agentic RAG 命令行入口，以及 FastAPI + Streamlit 本地 Web Demo。Agent 可使用 `knowledge_search` 和 `list_knowledge_sources` 工具，按 Tool Schema 校验参数，并在单个问题中最多进行六轮工具调用；会话保留最近六轮问答。

## 功能

- Markdown 知识文档按二级标题切分；Web 上传支持 Markdown 和 UTF-8 TXT，TXT 按段落及最大块长拆分。
- SiliconFlow BGE-M3 Embedding 与本地持久化 Chroma。
- 标题关键词优先结合向量相似度的混合检索。
- DeepSeek grounded answer generation、工具选择和工具结果迭代。
- Tool Schema、Registry、参数校验、Executor 与 Agent Loop。
- FastAPI 提供健康检查、Agent/传统 RAG 问答、SSE 流式回答、上传、列表、原文预览和会话清理接口。
- Streamlit 使用暗色主题、自定义 Hero 标题和文档卡片；新会话提供 RAG 基础、混合检索、RAG 搭建流程及 Agentic RAG 预设问题，点击即可进入现有问答流程。
- Streamlit 左侧提供文档上传与只读列表、Agent/RAG 模式开关；点击文档名后，原文在主内容区预览。聊天支持流式回答、清空会话，并在 `st.status` 中展示 Agent 工具、参数和检索片段数量。
- 回答引用以可展开面板展示来源和实际检索到的原始 chunk，方便查看答案依据。
- 上传限制文件类型、UTF-8 内容、文件大小及文件名；同名上传按文档更新处理，并清理旧向量。
- 按 session UUID 隔离对话历史；历史保存在 API 进程内，最长保留 24 小时，服务重启后清空。
- 原有 `rag_qa.py` 和 `agent_qa.py` 命令行入口保留。

## 架构

```text
Browser → Streamlit (8501) → FastAPI (8000)
                                  ├─ KnowledgeService
                                  ├─ Agent / Tool Registry / Executor
                                  ├─ Hybrid Retriever → Chroma (persistent)
                                  └─ SiliconFlow BGE-M3 + DeepSeek APIs
```

Web 知识库包含 `data/documents/` 中的内置 Markdown 文件和用户上传文档。上传文件存放在 `data/uploads/`，上传清单记录片段 ID；Chroma 和上传文件均为本地持久化数据，默认路径可通过环境变量修改。当前 Web API 只提供文档列表和原文预览，没有文档删除接口。

Agent 提供两个工具：`knowledge_search` 进行混合检索，`list_knowledge_sources` 查看可用文档来源。Web UI 会展示当轮工具调用和检索来源。引用来源以文档名显示，不向 Web 响应暴露服务器绝对路径。

### Web UI 交互

- 空白会话显示四个预设问题；点击问题会直接发起 Agent 或传统 RAG 问答。开始对话后预设问题隐藏，清空会话后重新显示。
- Agent 模式通过状态面板逐项展示工具调用名称、参数、检索到的片段数和来源；传统 RAG 模式不显示 Agent 工具轨迹。
- 回答下方的引用面板可以展开查看来源文件名和原始检索片段；文档列表保持只读，不提供文档删除操作。
- 页面采用深色配色和自定义样式，聊天输入框支持直接提问；也可点击左侧文档，在主内容区查看原文。

## 需求与安装

需要 **Python 3.12**。在仓库根目录运行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，填写你自己的服务密钥：

```dotenv
SILICONFLOW_API_KEY=your_siliconflow_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

真实 Key 只从环境变量或本地 `.env` 读取。`.env` 和 `data/uploads/`、Chroma 数据目录均被 `.gitignore` 忽略；请勿提交密钥。可选配置：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHROMA_DB_PATH` | `./chroma_db` | Chroma 持久化目录 |
| `UPLOADS_DIR` | `data/uploads` | 上传原文和索引清单目录 |
| `MAX_UPLOAD_SIZE_MB` | `5` | 单文件上限 |
| `MAX_QUESTION_LENGTH` | `2000` | 问题字符数上限 |
| `CHAT_REQUESTS_PER_MINUTE` | `20` | 每个客户端 IP 与 Session ID 组合每分钟问答上限（单进程内计数） |
| `UPLOADS_PER_MINUTE` | `5` | 每个客户端 IP 每分钟上传请求上限（单进程内计数） |

## 启动 Web Demo

在项目根目录打开两个 PowerShell 终端并激活同一虚拟环境。

终端一启动 API：

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

终端二启动 Streamlit：

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

浏览器打开 Streamlit 显示的本地地址（通常为 `http://localhost:8501`）。API 仅绑定本机回环地址；首次上传或问答会调用外部 Embedding/LLM API，可能产生相应的第三方 API 用量。Swagger 接口文档可在 API 启动后通过 `http://127.0.0.1:8000/docs` 查看。

API 响应采用统一结构：

```json
{"success": true, "message": "ok", "data": {}, "error": null}
```

错误响应设置相应 HTTP 状态码，包含 `success: false` 和可读错误信息；服务端异常不会向响应返回 SDK 错误内容或密钥。

### API 路由

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `POST` | `/chat` | JSON 问答，需 `session_id` 与 `question`，`mode` 可选 `agent`/`rag` |
| `POST` | `/chat/stream` | SSE 流式问答，支持 Agent 工具过程事件及普通 RAG 模式 |
| `POST` | `/documents/upload` | 上传 Markdown 或 TXT |
| `GET` | `/documents` | 查看内置和上传的文档 |
| `GET` | `/documents/{document_id}` | 读取文档原文用于页面预览 |
| `DELETE` | `/sessions/{session_id}` | 清空指定会话 |

API 的频率限制和会话存储均在单进程内；服务重启会清空对话历史。当前版本是本地演示应用，没有用户登录/身份认证，不应直接暴露到公网。公开部署前需要加访问控制，并将限流替换为可共享的存储/网关策略；公网部署和云资源不在本阶段范围内。

## CLI 使用

在 `data/documents/` 放入 Markdown 文档后运行：

```powershell
python rag_qa.py
```

传统 RAG 只检索并生成回答。Agentic RAG 命令行入口：

```powershell
python agent_qa.py
```

两种 CLI 入口继续使用相同的知识库、Embedding、Chroma、Retriever 和 DeepSeek 配置。CLI 读取知识库 Markdown 文件；在线上传管理在 Web 应用中执行。

## 测试

离线单元测试（默认排除真实外部 API 集成测试）：

```powershell
python -m pytest
```

覆盖率：

```powershell
python -m pytest --cov=src --cov=api --cov=app --cov-report=term-missing --cov-report=xml
```

真实服务集成测试需有效的 SiliconFlow/DeepSeek API Key、网络和服务可用性：

```powershell
python -m pytest -o addopts= -m integration
```

集成测试可能产生第三方 API 用量。GitHub Actions 使用 Python 3.12，执行离线测试，并上传 `coverage.xml`。工作流结果请查看上方徽章；README 不预设远程 CI 当前状态。

## 当前知识库文档

`data/documents/` 当前包含：

- `agent_intro.md`：Agent、RAG、Embedding、向量数据库和执行循环基础。
- `rag_intro.md`：RAG 定义、基本流程、优点和局限。
- `rag_build_intro.md`：RAG 搭建、切分、向量化、存储、检索、生成与评估。
- `agentic_rag_build_intro.md`：Agentic RAG 流程、工具调用和评估。
