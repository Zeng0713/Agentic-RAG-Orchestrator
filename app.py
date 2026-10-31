"""Streamlit UI for the Agentic RAG Orchestrator."""

import os
import json
import uuid

import requests
import streamlit as st


API_BASE_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = (3, 180)


def api_request(method, path, **kwargs):
    try:
        response = requests.request(
            method,
            f"{API_BASE_URL}{path}",
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
        try:
            payload = response.json()
        except ValueError:
            return None, "API 返回了无法识别的响应"
        if not response.ok or not payload.get("success"):
            return None, payload.get("error") or "请求失败"
        return payload.get("data"), None
    except requests.RequestException:
        return None, "无法连接 RAG API。请确认后端已启动。"


def refresh_documents():
    documents, error = api_request("GET", "/documents")
    if error:
        st.sidebar.error(error)
        return
    st.session_state.documents = documents or []


def stream_chat(session_id, question, mode, metadata, on_tool_event=None):
    """Read SSE events, yielding answer tokens for Streamlit's typewriter UI."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/stream",
            json={"session_id": session_id, "question": question, "mode": mode},
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )
        if not response.ok:
            try:
                payload = response.json()
                metadata["error"] = payload.get("error") or "请求失败"
            except ValueError:
                metadata["error"] = "API 返回了无法识别的响应"
            return

        for raw_line in response.iter_lines(decode_unicode=True):
            if isinstance(raw_line, bytes):
                raw_line = raw_line.decode("utf-8")
            if not raw_line or not raw_line.startswith("data: "):
                continue
            try:
                event = json.loads(raw_line[6:])
            except json.JSONDecodeError:
                metadata["error"] = "API 流式响应格式无效"
                return
            if event.get("type") == "token":
                yield event.get("text", "")
            elif event.get("type") == "tool":
                metadata["tool_events"].append(event)
                if on_tool_event:
                    on_tool_event(event)
            elif event.get("type") == "done":
                metadata["sources"] = event.get("sources", [])
                metadata["retrieved_chunks"] = event.get("chunks", [])
                metadata["mode"] = event.get("mode", mode)
            elif event.get("type") == "error":
                metadata["error"] = event.get("message", "服务暂时不可用")
                return
    except requests.RequestException:
        metadata["error"] = "无法连接 RAG API。请确认后端已启动。"
    finally:
        if "response" in locals():
            response.close()


def render_sources(sources, retrieved_chunks):
    if not sources and not retrieved_chunks:
        return
    with st.expander(f"📚 引用来源与检索片段（{len(retrieved_chunks)}）"):
        if sources:
            tags = " · ".join(f"`{source}`" for source in sources)
            st.markdown(f"**引用来源：** {tags}")
        for index, chunk in enumerate(retrieved_chunks, start=1):
            st.markdown(f"**资料 {index}：`{chunk['source']}`**")
            st.code(chunk["content"], language="markdown", wrap_lines=True)


def render_tool_events(tool_events):
    if not tool_events:
        return
    with st.status("🧭 Agent 工具调用轨迹", state="complete", expanded=False) as status:
        for index, event in enumerate(tool_events, start=1):
            tool_name = event.get("tool", "unknown")
            label = {
                "knowledge_search": "知识库检索",
                "list_knowledge_sources": "查看知识库来源",
            }.get(tool_name, tool_name)
            arguments = event.get("arguments", {})
            result = event.get("result", [])
            result_count = event.get("result_count", len(result) if isinstance(result, list) else 0)
            sources = event.get("sources") or [
                item.get("source") for item in result
                if isinstance(item, dict) and item.get("source")
            ]
            count_label = "返回来源数" if tool_name == "list_knowledge_sources" else "返回片段数"
            status.markdown(f"**{index}. {label}**")
            status.markdown("- 参数：")
            status.code(json.dumps(arguments, ensure_ascii=False, indent=2), language="json")
            status.markdown(f"- {count_label}：**{result_count}**")
            if sources:
                status.caption("来源：" + " · ".join(f"`{source}`" for source in sources))


st.set_page_config(
    page_title="Agentic RAG Orchestrator",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --rag-accent: #6366F1;
        --rag-bg: #0E1117;
        --rag-surface: #171B26;
        --rag-text: #E2E8F0;
        --rag-muted: #94A3B8;
    }

    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu,
    footer {
        visibility: hidden;
        height: 0;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 50% -15%, rgba(99, 102, 241, .18), transparent 42%),
            var(--rag-bg);
        color: var(--rag-text);
    }

    [data-testid="stSidebar"] > div:first-child {
        background: rgba(20, 24, 36, .78);
        border-right: 1px solid rgba(148, 163, 184, .14);
        box-shadow: 8px 0 30px rgba(0, 0, 0, .22);
        backdrop-filter: blur(16px);
    }

    .hero-section {
        text-align: center;
        padding: 2.1rem 1rem 1.7rem;
        margin: 0 auto 1.5rem;
    }
    .hero-title {
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: -.045em;
        background: linear-gradient(100deg, #A5B4FC 10%, #818CF8 48%, #C084FC 92%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .hero-subtitle {
        color: var(--rag-muted);
        margin: .8rem 0 0;
        font-size: 1.02rem;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid rgba(148, 163, 184, .12);
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin: .8rem 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, .16);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(49, 76, 130, .58), rgba(42, 51, 76, .78));
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(30, 35, 47, .82);
    }

    [data-testid="stSidebar"] [data-testid="stButton"] button,
    [data-testid="stMain"] [data-testid="stButton"] button {
        border-radius: 10px;
        border: 1px solid rgba(129, 140, 248, .32);
        transition: transform .18s ease, box-shadow .2s ease, background .2s ease;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover,
    [data-testid="stMain"] [data-testid="stButton"] button:hover {
        background: linear-gradient(100deg, #4F46E5, #7C3AED);
        border-color: transparent;
        box-shadow: 0 6px 18px rgba(99, 102, 241, .28);
        transform: translateY(-1px);
    }

    [class*="st-key-document-card-"] {
        background: #1E232F;
        border: 1px solid rgba(148, 163, 184, .1);
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
    }
    [class*="st-key-document-card-"] [data-testid="stButton"] button {
        background: transparent;
        border-color: transparent;
        text-align: left;
    }
    [class*="st-key-prompt-card-"] [data-testid="stButton"] button {
        min-height: 88px;
        white-space: normal;
        text-align: left;
        justify-content: flex-start;
        background: linear-gradient(145deg, rgba(30, 35, 47, .94), rgba(24, 29, 42, .96));
        border: 1px solid rgba(129, 140, 248, .2);
        color: var(--rag-text);
        padding: 1rem 1.1rem;
    }
    [class*="st-key-prompt-card-"] [data-testid="stButton"] button:hover {
        background: linear-gradient(110deg, rgba(79, 70, 229, .42), rgba(124, 58, 237, .3));
        border-color: rgba(165, 180, 252, .65);
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "document_previews" not in st.session_state:
    st.session_state.document_previews = {}
if "selected_document_id" not in st.session_state:
    st.session_state.selected_document_id = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

st.markdown(
    """
    <section class="hero-section">
      <h1 class="hero-title">Agentic RAG Orchestrator</h1>
      <p class="hero-subtitle">基于 Hybrid Search 与 DeepSeek 的智能问答</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("知识库")
    agent_mode = st.toggle(
        "启用 Agent 模式",
        value=True,
        help="开启后由 Agent 决定是否调用知识搜索工具；关闭后使用传统 RAG 检索并生成回答。",
    )
    if st.session_state.pop("reset_uploader", False):
        st.session_state["document_uploader"] = None
    uploaded_file = st.file_uploader(
        "上传 Markdown 或 TXT 文档",
        type=["md", "txt"],
        help="单个文件大小上限由后端配置，默认 5 MB。",
        key="document_uploader",
    )
    if uploaded_file is not None and st.button("上传并建立索引", use_container_width=True):
        with st.spinner("正在解析并向量化文档，请稍候…"):
            result, error = api_request(
                "POST",
                "/documents/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())},
            )
        if error:
            st.error(error)
        else:
            state = "更新" if result.get("updated") else "新增"
            st.toast(f"{state}完成：{result['filename']}（{result['chunk_count']} 个片段）")
            st.session_state.document_previews.pop(result["document_id"], None)
            refresh_documents()
            st.session_state["reset_uploader"] = True
            st.rerun()

    st.subheader("已收录文档")
    if st.button("刷新列表", use_container_width=True):
        refresh_documents()
    if not st.session_state.documents:
        refresh_documents()
    if not st.session_state.documents:
        st.caption("知识库暂时没有文档。")
    for document in st.session_state.documents:
        document_id = document["document_id"]
        with st.container(key=f"document-card-{document_id}"):
            if st.button(
                f"📄 {document['filename']} · {document['chunk_count']} chunks",
                key=f"open-{document_id}",
                use_container_width=True,
            ):
                st.session_state.selected_document_id = (
                    None if st.session_state.selected_document_id == document_id else document_id
                )
            st.caption(document["origin"])

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        _, error = api_request("DELETE", f"/sessions/{st.session_state.session_id}")
        if error:
            st.error(error)
        else:
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.success("会话已清空")
            st.rerun()

selected_document_id = st.session_state.selected_document_id
if selected_document_id:
    selected_document = next(
        (
            document for document in st.session_state.documents
            if document["document_id"] == selected_document_id
        ),
        None,
    )
    if selected_document is None:
        st.session_state.selected_document_id = None
    else:
        preview = st.session_state.document_previews.get(selected_document_id)
        if preview is None:
            with st.spinner("正在读取文档…"):
                preview, error = api_request(
                    "GET",
                    f"/documents/{selected_document_id}",
                )
            if error:
                st.error(error)
            else:
                st.session_state.document_previews[selected_document_id] = preview
        if preview is not None:
            with st.container(border=True):
                st.subheader(f"文档预览：{preview['filename']}")
                st.caption("点击左侧已选中的文档名可收起预览。")
                language = "markdown" if preview["file_type"] == "md" else "text"
                st.code(preview["content"], language=language, wrap_lines=True)
            st.divider()

if not st.session_state.messages and not st.session_state.pending_question:
    st.markdown("#### ✨ 试试这样提问")
    st.caption("选择一个问题，Agent 会直接基于当前知识库为你检索并回答。")
    suggested_questions = [
        "RAG 是什么？它的基本流程有哪些？",
        "混合检索是什么？关键词检索和向量检索如何互补？",
        "从文档上传到向量入库，RAG 搭建要经过哪些步骤？",
        "Agentic RAG 与传统 RAG 有什么区别？",
    ]
    for row_start in range(0, len(suggested_questions), 2):
        columns = st.columns(2)
        for offset, column in enumerate(columns):
            question_index = row_start + offset
            if question_index >= len(suggested_questions):
                continue
            prompt_text = suggested_questions[question_index]
            with column, st.container(key=f"prompt-card-{question_index}"):
                if st.button(
                    prompt_text,
                    key=f"suggested-question-{question_index}",
                    use_container_width=True,
                ):
                    st.session_state.pending_question = prompt_text
                    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(
            message.get("sources", []),
            message.get("retrieved_chunks", []),
        )
        if message.get("mode", "agent") == "agent":
            render_tool_events(message.get("tool_events", []))

question = st.chat_input("在这里输入你的问题...") or st.session_state.pop("pending_question", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        mode = "agent" if agent_mode else "rag"
        metadata = {
            "tool_events": [],
            "sources": [],
            "retrieved_chunks": [],
            "error": None,
            "mode": mode,
        }

        status = (
            st.status("Agent 正在分析并调用工具…", expanded=True)
            if agent_mode else None
        )

        def log_tool_event(event):
            if status is None:
                return
            tool_name = event.get("tool", "unknown")
            label = {
                "knowledge_search": "知识库检索",
                "list_knowledge_sources": "查看知识库来源",
            }.get(tool_name, tool_name)
            count_label = "返回来源数" if tool_name == "list_knowledge_sources" else "检索到片段数"
            status.markdown(f"**调用工具：{label}**")
            status.markdown(f"- 参数：`{json.dumps(event.get('arguments', {}), ensure_ascii=False)}`")
            status.markdown(f"- {count_label}：**{event.get('result_count', 0)}**")
            if event.get("sources"):
                status.caption("来源：" + " · ".join(f"`{source}`" for source in event["sources"]))

        answer = st.write_stream(
            stream_chat(
                st.session_state.session_id,
                question,
                mode,
                metadata,
                on_tool_event=log_tool_event,
            )
        )
        if metadata["error"]:
            if status is not None:
                status.update(label="Agent 调用失败", state="error")
            st.error(metadata["error"])
        else:
            if status is not None:
                status.update(label="Agent 调用完成", state="complete", expanded=True)
            render_sources(metadata["sources"], metadata["retrieved_chunks"])
            mode_label = "Agent" if metadata["mode"] == "agent" else "传统 RAG"
            st.caption(f"回答模式：{mode_label}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": metadata["sources"],
                "retrieved_chunks": metadata["retrieved_chunks"],
                "tool_events": metadata["tool_events"],
                "mode": metadata["mode"],
            })
