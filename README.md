[![Tests](https://github.com/Zeng0713/Agentic-RAG-Orchestrator/actions/workflows/tests.yml/badge.svg)](https://github.com/Zeng0713/Agentic-RAG-Orchestrator/actions/workflows/tests.yml)

# Agentic RAG Orchestrator

*A Modular and Agentic RAG Engine with Hybrid Retrieval, Chroma, and DeepSeek Tool Calling.*

基于 **LangChain、Chroma、BGE-M3 与 DeepSeek** 的本地知识库问答应用。通过桌面启动器打开知识工作台，上传文档、进行多轮问答，并查看回答依据和工具执行过程；同时提供 FastAPI 接口和传统 RAG / Agentic RAG 两个 CLI 入口。

## 快速开始：Windows 桌面启动

需要 **Python 3.12**、可访问模型服务的网络，以及 DeepSeek / SiliconFlow API Key。

1. 双击仓库根目录的 **`安装依赖.bat`**。脚本优先复用 `.venv` 或 `venv`；没有虚拟环境时使用 Python 3.12 创建 `.venv`，安装依赖，并在 `.env` 不存在时复制 `.env.example`。
2. 在本地 `.env` 中填写自己的密钥：

   ```dotenv
   DEEPSEEK_API_KEY=your_deepseek_api_key
   SILICONFLOW_API_KEY=your_siliconflow_api_key
   ```

3. 双击 **`创建桌面快捷方式.vbs`**，生成 **Agentic RAG Orchestrator** 桌面图标。
4. 双击桌面图标，或直接双击项目中的 **`启动项目.vbs`**。启动器检查配置、启动后端和前端，等待就绪后打开独立工作台窗口。

日常使用无需分别打开两个终端。启动器提供“打开工作台”“查看日志”“停止服务”和“重新启动”。关闭工作台窗口只关闭页面；关闭启动器会停止它创建的服务和应用窗口。重复打开快捷方式会唤起已有启动器。

默认 **Memory 模式不需要 Docker 或 Redis**。启用 Redis 后，启动器先检查已有连接；仅对默认本机 Redis 配置尝试使用 Docker Desktop 和项目 Compose 文件启动。连接失败时询问是否仅本次改用 Memory，不修改 `.env`，也不迁移 Redis 历史。已有后端的状态配置保持不变。

启动器会核对服务的项目目录、进程命令和健康状态，安全复用已运行的项目服务；复用服务不接管原日志，也不在退出时停止。遇到其他程序占用端口会提示处理，不自动结束该程序。Docker Desktop、Redis 容器及持久化数据在退出后保留。

独立窗口优先使用已安装的 Edge / Chrome，找不到时使用默认浏览器。这是依赖本机 Python 环境的桌面启动方式，并非免安装 EXE；移动项目目录后需重新创建快捷方式。

## 核心能力

| 能力 | 实现 |
|---|---|
| 双模式问答 | 经典 RAG 直接检索并生成；Agentic RAG 自主选择工具、读取结果并迭代 |
| 混合检索 | 标题关键词优先结合向量相似度；SiliconFlow BGE-M3 向量化，本地 Chroma 持久化 |
| 工具编排 | DeepSeek Tool Calling，Tool Schema → Registry → 参数校验 / Executor → Agent Loop；单题最多六轮工具决策 |
| 多轮对话 | Web 和 Agent CLI 保留最近六轮完整问答；Web 普通 RAG 结合历史改写追问、检索后生成 |
| 文档管理 | Markdown / UTF-8 TXT 上传、解析、切分、入库、列表和原文预览；文档列表只读，无删除功能 |
| 回答可追溯 | SSE 流式输出，实时展示工具、参数和片段数量；弹窗查看引用片段与工具记录 |
| 提问引导 | 首页预设问题；回答后基于问题、回答、检索片段和实际文档推荐最多三个相关追问，不额外调用模型 |
| 状态与限流 | 按 Session UUID 隔离对话；Memory / Redis 可分别配置会话与滑动窗口限流 |
| 本地运行 | Windows 桌面启动器、自动就绪检查、有界日志；保留手动 Web 和 CLI 启动 |

网页采用深色知识工作台，侧栏支持文档筛选、导入、模式切换和清空对话，原文在主区域阅读。对话区域独立滚动；引用和工具记录在可滚动弹窗中查看。生成期间锁定请求模式并禁用会触发重跑的操作，避免切换模式中断回答；完成或失败后恢复。

上传校验包括扩展名、文件名、大小、空内容、UTF-8 编码和明显二进制内容；使用生成的存储名避免路径穿越。同名上传按更新处理，成功发布后清理旧向量和旧版本原文，失败时尝试补偿回滚。应用层补偿不等同于跨文件和向量库的严格事务。

Markdown 按二级标题切分；TXT 按段落与最大块长拆分。CLI 启动或 API 首次初始化知识库时，同步完整内置文档集合，处理新增、更新、删除或缩短章节产生的旧片段，保留上传文档及无法确认归属的记录。修改 `data/documents/` 后，重启 API 并访问知识库接口，或重新运行 CLI 才会同步。

## 架构与目录

```text
桌面启动器 ── 启停 / 就绪检查 / 日志 ──┐
浏览器 / 独立工作台 → Streamlit → FastAPI
                                      ├─ RateLimiter → Memory / Redis
                                      └─ KnowledgeService ← SessionStore → Memory / Redis
                                           ├─ 经典 RAG / Agent Loop
                                           ├─ Tool Schema → Registry → Executor
                                           ├─ Hybrid Retriever → Chroma
                                           └─ SiliconFlow BGE-M3 / DeepSeek APIs
```

Agent 的两个工具：`knowledge_search` 检索知识片段；`list_knowledge_sources` 列出知识库来源。工具结果和模型上下文有长度预算，引用片段可能被截断，完整内容可通过原文预览查看。Web 响应使用文件名展示来源，不暴露服务器绝对路径。

```text
Agentic-RAG-Orchestrator/
├── launcher.pyw / desktop_launcher/   # 原生启动器、进程管理、日志轮转
├── 启动项目.vbs / 创建桌面快捷方式.vbs / 安装依赖.bat
├── start.ps1                         # PowerShell 启动入口
├── app.py / assets/ / .streamlit/     # Streamlit、图标与主题
├── api/main.py                       # FastAPI 与 SSE
├── src/                              # RAG、Agent、工具、状态与知识库服务
├── data/documents/                   # 内置 Markdown
├── data/uploads/                     # 上传原文和清单（运行时生成）
├── chroma_db/                        # 持久化向量（运行时生成）
├── logs/ / .runtime/                  # 日志、实例记录和应用窗口配置（忽略提交）
├── rag_qa.py / agent_qa.py            # CLI
├── tests/ / docs/
├── compose.redis.yml
└── .github/workflows/tests.yml
```

## 配置与数据

真实 Key 只从环境变量或本地 `.env` 读取，不写入前端、代码或 Git。进程环境变量优先于 `.env`。`.env`、上传数据、默认 Chroma 目录、日志与运行时浏览器配置已被 Git 忽略。完整配置模板见 [`.env.example`](.env.example)。

| 变量 | 默认值 | 用途 |
|---|---|---|
| `CHROMA_DB_PATH` | `./chroma_db` | Chroma 持久化目录 |
| `UPLOADS_DIR` | `data/uploads` | 上传原文和索引清单 |
| `MAX_UPLOAD_SIZE_MB` | `5` | 单文件大小上限 |
| `MAX_QUESTION_LENGTH` | `2000` | 问题字符数上限 |
| `CHAT_REQUESTS_PER_MINUTE` | `20` | 每 IP + Session 的问答额度 |
| `CHAT_IP_REQUESTS_PER_MINUTE` | `60` | 每 IP 的问答总额度 |
| `UPLOADS_PER_MINUTE` | `5` | 每 IP 的上传额度 |
| `SESSION_BACKEND` / `RATE_LIMIT_BACKEND` | `memory` | 分别选择 `memory` 或 `redis` |
| `LAUNCHER_API_PORT` / `LAUNCHER_UI_PORT` | `8000` / `8501` | 桌面启动器使用的端口 |
| `LOG_RETENTION_DAYS` | `7` | 启动器日志保留天数 |
| `LOG_MAX_SIZE_MB` / `LOG_MAX_FILES` | `20` / `6` | 每类日志单文件大小及总文件数（包含当前文件） |

### 日志

桌面启动生成项目根目录下的 `logs/launcher.log`、`logs/backend.log` 和 `logs/frontend.log`。默认每类最多 **6 个文件，单文件 20 MB**，三类合计约 **360 MB** 上限；按大小及 UTC 日期轮转，超过 7 天的日志会清理。启动、写入和启动器运行期间每分钟触发清理；启动器退出后不驻留清理任务，下次启动继续清理。

服务日志仅保存允许的访问状态、执行状态、耗时和错误类型等元数据，并过滤密钥及 Redis 凭据，不持久化任意控制台输出、聊天正文或完整 SDK 堆栈。复用手动启动的服务时保留其原有日志方式；手动 CLI / Web 命令本身不会启用启动器的文件轮转机制。

### 状态存储边界

| 状态 | 存储位置 | 共享与生命周期 |
|---|---|---|
| Web 最近六轮问答、会话锁 | Memory / Redis | Memory 随 API 退出清空；Redis 支持跨进程共享，历史默认闲置 24 小时过期 |
| IP、IP + Session 滑动窗口计数 | Memory / Redis | Redis 模式合并不同后端进程的额度 |
| CLI 聊天历史 | CLI 进程内存 | 不与 Web 共享 |
| 内置文档、上传原文与清单 | 本地文件 | 正常重启保留，属于本地知识库存储 |
| 文档向量、运行时文档视图 | 本地 Chroma / 进程内存 | 向量持久化，视图随进程加载；没有跨实例更新协调 |

Redis 共享的是会话和限流状态。文档、清单、Chroma 仍为本地单点，不能让多个独立写入进程直接共享这些本地数据，也不代表应用已支持完整多机器部署。

## Redis（可选）

在 `.env` 中启用需要共享的组件，重启后端：

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

也可以在仓库根目录手动管理本机 Redis：

```powershell
docker compose -f compose.redis.yml up -d
docker compose -f compose.redis.yml stop
```

示例容器仅绑定本机，启用 AOF 与独立数据卷；`stop` 保留数据卷，Redis 持久化不保证零丢失。外部可用 Redis 无需 Docker。

任一组件启用 Redis 后，URL 缺失、无效或启动连接检查失败都会阻止 API 启动；URL 不接受查询参数，避免绕过固定超时。运行期故障返回 503，不静默切回 Memory；Redis 恢复后客户端可重新连接。桌面启动器的显式 Memory 选择只影响它新建的后端进程。

### 设计决策

| 决策 | 行为与原因 |
|---|---|
| 同会话并发 | 锁覆盖读历史、生成、保存；重复请求立即 409。租约 60 秒，每 15 秒校验续租，失锁后禁止提交，避免乱序覆盖。 |
| 生成成功、历史保存失败 | 按 request ID 幂等提交，暂时故障最多三次尝试；仍无法确认时返回答案及 `history_status=unconfirmed`，避免重复生成。 |
| 限流 | 60 秒滑动窗口；Redis 在同一 Lua 脚本判断 IP + Session 与 IP 总额度，防止换 Session 绕过；429 带 `Retry-After`。 |
| 后端选择 | 会话和限流分别配置，独立注入；切换后端不迁移已有内存会话。 |

会话提交用 Lua 校验锁、去重、追加完整问答、裁剪六轮并设置 TTL；原子执行不意味着脚本错误回滚或 Redis 故障切换强一致。知识库锁只保护检索视图和更新发布，Embedding 准备与 LLM 生成不占知识库写锁，发布时检索可短暂等待。实现与故障验证详见 [Redis 设计文档](docs/redis_design.md)。

## 手动运行与 API

开发或非 Windows 环境可使用原有命令。在项目根目录创建 Python 3.12 虚拟环境、安装依赖并配置 `.env`：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

分别在两个终端运行：

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

```powershell
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

已有 `venv` 时使用对应解释器；Linux / macOS 将解释器路径替换为 `.venv/bin/python`。前端默认连接 `http://127.0.0.1:8000`，自定义后端地址使用进程环境变量 `RAG_API_URL`。浏览器打开 `http://127.0.0.1:8501`，Swagger 文档位于 `http://127.0.0.1:8000/docs`。

网页首次读取文档列表会初始化知识库；新增或变化的片段需要调用 Embedding，上传与问答按需调用 Embedding / LLM，可能产生第三方 API 用量。启动器的就绪探测本身不调用模型 API。

| 方法 | 路径 | 功能 |
|---|---|---|
| GET | `/health` | 进程存活 |
| GET | `/ready` | 会话 / 限流依赖就绪检查，不验证模型或完整问答链路 |
| POST | `/chat` | JSON 问答：`session_id`、`question`，可选 `mode=agent/rag` |
| POST | `/chat/stream` | SSE token、工具、完成及错误事件 |
| POST | `/documents/upload` | 上传 Markdown / TXT |
| GET | `/documents` | 文档列表 |
| GET | `/documents/{document_id}` | 原文预览 |
| DELETE | `/sessions/{session_id}` | 清空指定会话 |

JSON 接口统一返回 `{"success":true,"message":"ok","data":{},"error":null}`；错误使用相应 HTTP 状态码和清晰错误信息，不返回 SDK 原始异常或密钥。SSE 开始前完成限流、会话锁及历史读取，开始后以 `error` 事件报告故障；只有 `done` 表示本轮完成，前端会提示中断和历史保存警告。

CLI 使用相同的知识库核心和模型配置：

```powershell
python rag_qa.py     # 传统 RAG，单轮问答
python agent_qa.py   # Agentic RAG，保留最近六轮问答
```

CLI 读取内置 Markdown；上传管理通过 Web 执行。需先激活相应虚拟环境，或使用其 Python 完整路径运行。

## 测试与 CI

离线测试使用 Mock，默认排除四项真实模型 API 集成测试，并拦截未 Mock 的模型调用：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
python -m pytest --cov=src --cov=api --cov=app --cov=desktop_launcher --cov-report=term-missing --cov-report=xml
```

真实 Redis 跨进程测试不调用模型；未设置 `TEST_REDIS_URL` 时跳过：

```powershell
$env:TEST_REDIS_URL="redis://127.0.0.1:6379/0"
python -m pytest tests/test_redis_processes.py -q
```

容器停止 / 恢复故障注入需专用测试容器，具体步骤见设计文档。真实模型集成测试需要有效密钥和网络，会消耗第三方 API 用量：

```powershell
python -m pytest -o addopts= -m integration
```

自动化行覆盖率：`src + api + app` 为 **83.87%（1388 / 1655）**；包含新增桌面启动器的完整统计为 **68.74%（1658 / 2412）**。原生窗口、浏览器启动和部分 Windows 进程分支尚未被单元覆盖率覆盖；另外已实际验证隔离端口启动 / 停止、已有服务复用、原生窗口就绪 / 退出和单实例机制。浏览器检查覆盖桌面与窄屏顶部布局、图标和侧栏重新展开。上述独立冒烟检查及 Uvicorn 子进程未合并进覆盖率。

## 知识库内容

| 文件 | 内容 |
|---|---|
| `agent_intro.md` | Agent、RAG、Embedding、向量数据库和执行循环 |
| `rag_intro.md` | RAG 定义、流程、价值与局限 |
| `rag_build_intro.md` | 文档处理、切分、检索、评估和索引同步排查 |
| `agentic_rag_build_intro.md` | 与经典 RAG 的区别、工具编排、升级步骤与评估 |
| `retrieval_quality_guide.md` | 检索排查、Chunk、Top K、查询改写、评估与引用核对 |
| `project_operations_guide.md` | 本地运行、上传更新、持久化、状态码及故障处理 |

文档包含 BM25、RRF、Reranker 等扩展概念的介绍，不代表代码已实现这些算法。

## 使用与部署边界

- 当前定位为本地使用和面试演示，没有登录认证、租户知识库隔离或公网访问控制；Session UUID 用于对话隔离，不是权限凭证。
- Streamlit 由服务端转发 API 请求，后端通常看到 Streamlit 的 IP，多个浏览器可能共享 IP 额度。公开部署需要可信身份方案，不能直接信任任意转发头。
- 支持 Markdown 和 UTF-8 TXT，没有 PDF / Word 解析器；流式回答没有断点续传，刷新页面不会自动接回未完成请求。
- 备份应同时覆盖内置文档、上传原文及清单、Chroma 和必要配置，避免写入期间复制不一致的数据集合。更换 Embedding 模型需重新建索引，不能混用新旧向量。
- 部署服务器前需准备访问控制、HTTPS、进程托管、备份和资源评估；完整多实例部署还需要统一文档存储、共享向量服务与更新协调。
