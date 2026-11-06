[![Tests](https://github.com/Zeng0713/Agentic-RAG-Orchestrator/actions/workflows/tests.yml/badge.svg)](https://github.com/Zeng0713/Agentic-RAG-Orchestrator/actions/workflows/tests.yml)

# Agentic RAG Orchestrator

*A Modular and Agentic RAG Engine with Hybrid Retrieval, Chroma, and DeepSeek Tool Calling.*

基于 **LangChain、Chroma、BGE-M3 与 DeepSeek** 的模块化知识库问答项目，提供传统 RAG 与 Agentic RAG 命令行入口，以及 FastAPI + Streamlit 本地 Web Demo。Agent 可使用 `knowledge_search` 和 `list_knowledge_sources` 工具，按 Tool Schema 校验参数，并在单个问题中最多进行六轮工具调用决策；Web 和 Agent CLI 会话保留最近六轮问答，传统 RAG CLI 为单轮问答。

## 功能

- Markdown 知识文档按二级标题切分；Web 上传支持 Markdown 和 UTF-8 TXT，TXT 按段落及最大块长拆分。
- SiliconFlow BGE-M3 Embedding 与本地持久化 Chroma。
- CLI 启动或 API 首次初始化知识库时，以完整内置文档集合同步索引：新增、更新片段，并清理已删除文档或缩短章节留下的旧片段；保留 Web 上传文档和无法确认归属的记录。修改 `data/documents/` 后需重启 API 并访问知识库接口，或重新运行 CLI 才会同步。
- 标题关键词优先结合向量相似度的混合检索。
- DeepSeek grounded answer generation、工具选择和工具结果迭代。
- Tool Schema、Registry、参数校验、Executor 与 Agent Loop。
- FastAPI 提供健康检查、Agent/传统 RAG 问答、SSE 流式回答、上传、列表、原文预览和会话清理接口。
- Streamlit 使用深色知识研究台、浅橄榄色强调、中文文档导航和主题/文件名筛选；首页显示实际文档与片段数量、问答路径示意，以及 RAG 基础、混合检索、搭建流程和 Agentic RAG 预设问题。
- Streamlit 左侧提供文档上传与只读列表、Agent/RAG 模式开关；点击文档名后，原文在主内容区预览。聊天支持流式回答、清空会话，并在 `st.status` 中展示 Agent 工具、参数和检索片段数量。
- 回答引用以弹窗展示来源和检索片段，内容在窗口内独立滚动，方便查看答案依据；工具结果和模型上下文受长度预算限制，片段可能被截断，完整原文可通过文档预览查看。
- 上传限制文件类型、UTF-8 内容、文件大小及文件名；同名上传按文档更新处理，并清理旧向量。
- 按 session UUID 隔离最近六轮问答；会话存储和滑动窗口限流分别支持 Memory/Redis 注入。Redis 模式可跨进程共享状态，历史闲置 24 小时过期；默认 Memory 模式在 API 重启后清空。
- 同会话并发立即返回 409；Redis 会话锁覆盖读历史、生成和保存，支持自动续租、持有者校验与幂等提交。历史保存无法确认时保留答案并在页面提示。
- Web 普通 RAG 先结合历史改写追问再检索，生成时携带经过预算裁剪的历史；Agent 保留工具调用和六轮循环上限。
- 原有 `rag_qa.py` 和 `agent_qa.py` 命令行入口保留。

## 架构

```text
Browser → Streamlit (8501) → FastAPI (8000)
                                  ├─ RateLimiter → Memory / Redis
                                  ├─ KnowledgeService ← SessionStore (Memory / Redis)
                                  ├─ Agent / Tool Registry / Executor
                                  ├─ Hybrid Retriever → Chroma (persistent)
                                  └─ SiliconFlow BGE-M3 + DeepSeek APIs
```

Web 知识库包含 `data/documents/` 中的内置 Markdown 文件和用户上传文档。上传文件存放在 `data/uploads/`，上传清单记录片段 ID；Chroma 和上传文件均为本地持久化数据，默认路径可通过环境变量修改。当前 Web 文档 API 支持上传、列表和原文预览，没有文档删除接口。

Agent 提供两个工具：`knowledge_search` 进行混合检索，`list_knowledge_sources` 查看可用文档来源。Web UI 会展示当轮工具调用和检索来源。引用来源以文档名显示，不向 Web 响应暴露服务器绝对路径。

### Web UI 交互

- 空白会话显示四个预设问题；点击问题会直接发起 Agent 或传统 RAG 问答。开始对话后预设问题隐藏，清空会话后重新显示。
- Agent 模式生成时通过状态面板展示工具调用名称、参数、检索片段数和来源；完成后可在“检索与工具调用”弹窗中回看。传统 RAG 模式不显示 Agent 工具轨迹。
- 回答生成期间锁定本次请求模式，暂时禁用模式切换、上传、文档操作、清空对话和重复提问，防止页面重跑打断流式回答；完成或失败后恢复操作。
- 成功回答后，在最新回答下方显示最多三个“继续探索”问题；基于当前问题、回答、实际检索片段和可用文档，在本地选择相关主题，过滤本次会话已问问题，不额外调用模型。点击后沿用当前 Session 和所选模式提问；无匹配主题或回答失败时不显示。
- 模式说明文字与开关分离，仅操作开关本体改变模式，点击说明文字和行内空白不切换。
- 回答下方的“引用来源与检索片段”和“检索与工具调用”按钮打开详情弹窗，窗口内滚动查看，关闭或按 Esc 返回对话；回答生成时仍使用状态面板显示实时工具过程。文档列表保持只读，不提供文档删除操作。
- 对话在独立阅读区域内滚动，关闭自动追底和滚动锚定；新内容不会强行改变用户阅读位置，长回答可手动滚动查看。
- 页面采用深色配色和自定义样式，聊天输入框支持直接提问；也可点击左侧文档，在主内容区切换阅读视图和原文。折叠侧栏后，左上角保留固定的展开入口；窄屏问题按钮支持换行。
- 侧栏以中文主题展示内置文档，支持按主题或原始文件名本地筛选（输入后按 Enter 或离开输入框确认）；未知上传文件保留文件名展示。首页数字来自当前 API 文档列表，问答路径面板是工作原理示意，不表示已执行检索。
- UI 基于 Streamlit 1.66.1，主题由 `.streamlit/config.toml` 管理，局部样式集中在 `assets/workspace.css`；升级 Streamlit 时需重新验证侧栏入口和样式选择器。

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
| `CHAT_REQUESTS_PER_MINUTE` | `20` | 每个客户端 IP 与 Session ID 组合每分钟问答上限 |
| `CHAT_IP_REQUESTS_PER_MINUTE` | `60` | 每个客户端 IP 的问答总额度，防止更换 Session 绕过细粒度限流 |
| `UPLOADS_PER_MINUTE` | `5` | 每个客户端 IP 每分钟上传请求上限 |

以上额度使用 60 秒滑动窗口；Memory 在进程内计数，Redis 模式跨进程合并计数。Streamlit 请求经服务端转发，API 当前看到的通常是 Streamlit 服务 IP；本地演示共享 IP 额度，公网部署前需设计可信客户端身份，不能直接信任任意转发头。

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

浏览器打开 Streamlit 显示的本地地址（通常为 `http://localhost:8501`）。API 仅绑定本机回环地址；网页首次加载文档列表也会触发知识库初始化，新增或变化的片段需要调用 Embedding API；上传和问答还会按需调用 Embedding/LLM API，可能产生第三方 API 用量。Swagger 接口文档可在 API 启动后通过 `http://127.0.0.1:8000/docs` 查看。

API 响应采用统一结构：

```json
{"success": true, "message": "ok", "data": {}, "error": null}
```

错误响应设置相应 HTTP 状态码，包含 `success: false` 和可读错误信息；服务端异常不会向响应返回 SDK 错误内容或密钥。

### API 路由

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `GET` | `/ready` | 检查配置的会话/限流依赖是否就绪，不主动调用模型 API |
| `POST` | `/chat` | JSON 问答，需 `session_id` 与 `question`，`mode` 可选 `agent`/`rag` |
| `POST` | `/chat/stream` | SSE 流式问答，支持 Agent 工具过程事件及普通 RAG 模式 |
| `POST` | `/documents/upload` | 上传 Markdown 或 TXT |
| `GET` | `/documents` | 查看内置和上传的文档 |
| `GET` | `/documents/{document_id}` | 读取文档原文用于页面预览 |
| `DELETE` | `/sessions/{session_id}` | 清空指定会话 |

API 默认使用进程内会话与限流，也可按下节切换 Redis。当前没有用户登录/身份认证，Session UUID 不等于权限控制。文档、上传清单及 Chroma 仍是本地单点；Redis 接入不代表整个应用已支持多机器部署。公网部署前还需访问控制和可信客户端身份；本阶段不配置公网服务。

## Redis 会话与限流（可选）

默认无需 Redis，原来的本地启动命令和 CLI 保持可用。Windows 已安装并启动 Docker Desktop 时，可在仓库根目录启动仅绑定本机的 Redis：

```powershell
docker compose -f compose.redis.yml up -d
```

然后在本地 `.env` 中设置并重启 API（不要提交真实连接凭据）：

```dotenv
SESSION_BACKEND=redis
RATE_LIMIT_BACKEND=redis
REDIS_URL=redis://127.0.0.1:6379/0
REDIS_KEY_PREFIX=rag
SESSION_TTL_SECONDS=86400
SESSION_LOCK_LEASE_SECONDS=60
SESSION_LOCK_RENEW_SECONDS=15
CHAT_DEADLINE_SECONDS=300
```

任一组件启用 Redis 后，缺少有效 URL 或启动连接检查失败都会阻止 API 启动；URL 不接受查询参数，避免绕过固定连接超时。运行期间不静默切回 Memory。Redis 恢复后客户端自动重连；Redis 重启后的历史保留取决于其持久化，示例启用 AOF 和独立数据卷，不保证零丢失。`docker compose -f compose.redis.yml stop` 停止服务并保留卷。

| 状态 | 存储位置 | 跨进程共享 |
|---|---|---|
| Web 最近六轮问答及会话锁 | Memory / Redis | Redis 模式支持 |
| IP、IP + Session 滑动窗口限流 | Memory / Redis | Redis 模式支持 |
| CLI 聊天历史 | CLI 进程内存 | 不与 Web 共享 |
| 内置文档、上传原文和清单 | 本地文件 | 本次不解决 |
| 文档向量、运行时文档视图 | 本地 Chroma / 内存 | 本次不解决 |

### 设计决策

| 取舍 | 推荐与理由 |
|---|---|
| 同会话并发 | 全程锁定读历史、生成和保存，重复请求立即 409，避免乱序和长时间排队；60 秒租约，每 15 秒校验并续租，失去锁后禁止提交。 |
| 答案生成成功、保存历史失败 | 按 request ID 幂等提交，暂时性故障最多三次尝试；仍无法确认则返回答案和 `history_status=unconfirmed` 警告，不重复生成。 |
| 限流维度 | IP + Session 与 IP 总额度在同一 Lua 脚本中判断，防止匿名切换 Session 绕过；超限返回 429 和 `Retry-After`。 |
| 配置开关 | Session/限流各有 Memory/Redis 开关，职责独立；两者都启用 Redis 才同时共享会话与额度。 |

会话提交以一轮完整问答为 List 元素，通过 Lua 校验锁、去重、追加、裁剪六轮并设置 TTL；原子执行不代表脚本错误自动回滚，也不保证 Redis 故障切换下的强一致。切换后端不迁移存量内存会话。普通 RAG 的追问改写与上下文裁剪是独立行为变更，详见 [Redis 设计与验证](docs/redis_design.md)。

知识库读写锁只保护检索视图和更新发布；Embedding 准备及 LLM 生成不占知识库写锁。更新发布时检索可短暂等待，未引入文档删除操作。多进程验证使用独立测试知识库，不能据此让多个生产写入进程直接共享本地 Chroma/JSON 清单。

运行期状态依赖故障返回统一错误体和 503，冲突返回 409。SSE 开始前完成限流、获取锁及读取历史；开始后故障使用 `error` 事件，`done` 才表示本轮完成。前端检测中断并展示历史保存警告。`/health` 是存活探针；`/ready` 检查状态依赖，不代表外部模型或整条知识库链路已经验证。

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
python -m pip install -r requirements-dev.txt
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

真实 Redis 测试不调用模型 API。未设置 `TEST_REDIS_URL` 时本地跳过，设置后会启动两个独立 Uvicorn 测试进程，使用临时知识库和独立 Redis 前缀：

```powershell
$env:TEST_REDIS_URL="redis://127.0.0.1:6379/0"
python -m pytest tests/test_redis_processes.py -q
```

CI 配置 Redis 服务容器来执行跨进程共享测试；Docker 停止/恢复故障注入需显式指定专用测试容器，见设计文档。普通离线测试拦截未 Mock 的模型调用，避免消耗 API 额度。远程工作流结果以 GitHub Actions 页面为准，本地测试通过不等同于远程 CI 成功。

本地验证快照（2026-11-06，包含最新 UI、知识库、真实 Redis 7.4 跨进程及故障恢复测试）：**120 passed，4 deselected；`src + api + app` 行覆盖率 83.73%（1384 / 1653）**。4 项真实模型 API 集成测试未运行；CI 未指定故障注入容器时会跳过停止/恢复容器这一项。独立 Uvicorn 子进程未合并到覆盖率统计，详见设计文档中的验证记录。

## 当前知识库文档

`data/documents/` 当前包含：

- `agent_intro.md`：Agent、RAG、Embedding、向量数据库和执行循环基础。
- `rag_intro.md`：RAG 定义、基本流程、优点和局限。
- `rag_build_intro.md`：从上传到向量入库的完整链路、切分、检索与评估，以及入库验证、索引旧数据排查和离线/在线阶段区别。
- `agentic_rag_build_intro.md`：Agentic RAG 与传统 RAG 的直接对比、升级步骤、工具与会话轮数区别，以及工具调用、选型和评估。
- `retrieval_quality_guide.md`：检索失败排查、Chunk 与元数据、BM25/RRF/Reranker 概念、Top K、查询改写、评估指标、引用核对与资料不足处理。概念介绍不等于项目已实现对应算法。
- `project_operations_guide.md`：本地启动、健康检查、上传更新、持久化、409/429/503、Redis 故障、历史保存半失败、流式中断和备份边界。
