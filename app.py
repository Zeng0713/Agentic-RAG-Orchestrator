"""Streamlit UI for the Agentic RAG Orchestrator."""
import os
import base64
import json
import uuid
from pathlib import Path

import requests
import streamlit as st
from src.followups import suggest_followups


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
    completed = False
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
                completed = True
                metadata["history_status"] = event.get("history_status", "saved")
                metadata["warning"] = event.get("warning")
                metadata["sources"] = event.get("sources", [])
                metadata["retrieved_chunks"] = event.get("chunks", [])
                metadata["mode"] = event.get("mode", mode)
            elif event.get("type") == "error":
                metadata["error"] = event.get("message", "服务暂时不可用")
                return
        if not completed:
            metadata["error"] = "回答连接提前中断，本轮未确认完成，请重试。"
    except requests.RequestException:
        metadata["error"] = "无法连接 RAG API。请确认后端已启动。"
    finally:
        if "response" in locals():
            response.close()


@st.dialog("引用来源与检索片段", width="large")
def show_sources(sources, retrieved_chunks):
    st.caption(f"{len(sources)} 个来源 · {len(retrieved_chunks)} 个检索片段")
    with st.container(height=420, border=False, key="source-window"):
        if sources:
            st.markdown(" · ".join(f"`{source}`" for source in sources))
        for index, chunk in enumerate(retrieved_chunks, start=1):
            st.markdown(f"**资料 {index}：`{chunk['source']}`**")
            st.code(chunk["content"], language="markdown", wrap_lines=True)


@st.dialog("检索与工具调用", width="large")
def show_tool_events(tool_events):
    st.caption(f"本轮共 {len(tool_events)} 次工具调用")
    with st.container(height=420, border=False, key="tool-window"):
        for index, event in enumerate(tool_events, start=1):
            tool_name = event.get("tool", "unknown")
            label = {"knowledge_search": "知识库检索", "list_knowledge_sources": "查看知识库来源"}.get(tool_name, tool_name)
            result = event.get("result", [])
            count = event.get("result_count", len(result) if isinstance(result, list) else 0)
            sources = event.get("sources") or [item.get("source") for item in result
                       if isinstance(item, dict) and item.get("source")]
            st.markdown(f"**{index}. {label}**")
            st.code(json.dumps(event.get("arguments", {}), ensure_ascii=False, indent=2), language="json")
            st.markdown(f"- 返回数量：**{count}**")
            if sources:
                st.caption("来源：" + " · ".join(f"`{source}`" for source in sources))


def render_sources(sources, retrieved_chunks, *, key):
    if (sources or retrieved_chunks) and st.button(
        f"引用来源与检索片段（{len(retrieved_chunks)}）", icon=":material/library_books:",
        key=key, width="stretch", disabled=st.session_state.get("is_generating", False),
    ):
        show_sources(sources, retrieved_chunks)


def render_tool_events(tool_events, *, key):
    if tool_events and st.button(
        "检索与工具调用", icon=":material/route:", key=key,
        width="stretch", disabled=st.session_state.get("is_generating", False),
    ):
        show_tool_events(tool_events)


st.set_page_config(
    page_title="Agentic RAG Orchestrator",
    page_icon=str(Path(__file__).parent / "assets" / "app-icon.png"),
    initial_sidebar_state="expanded",
    layout="wide",
)

st.html(Path(__file__).parent / "assets" / "workspace.css")

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

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
# A browser refresh may interrupt a consumed request; do not leave controls locked.
if st.session_state.is_generating and "pending_request" not in st.session_state:
    st.session_state.is_generating = False
busy = st.session_state.is_generating

with st.sidebar:
    with st.container(key="sidebar-brand"):
        brand_icon = base64.b64encode((Path(__file__).parent / "assets" / "app-icon.png").read_bytes()).decode("ascii")
        st.html(f'<div class="library-brand"><span class="brand-symbol" aria-hidden="true"><img src="data:image/png;base64,{brand_icon}" alt="" width="40" height="40"></span><div><strong>Agentic RAG</strong><span>你的知识研究台</span></div></div>')
    st.markdown("问答方式")
    with st.container(horizontal=True, vertical_alignment="center", gap="small", key="mode-switch"):
        agent_mode = st.toggle(
            "启用 Agent 模式",
            value=True,
            disabled=busy,
            label_visibility="collapsed",
            width="content",
        )
        st.markdown("启用 Agent 模式", width="content")
    if not busy:
        st.caption("自主选择工具，逐步检索资料" if agent_mode else "检索相关片段后直接生成回答")
    if busy:
        st.caption("正在生成回答，完成后可切换模式。")
    with st.expander("导入文档", icon=":material/upload_file:"):
        if st.session_state.pop("reset_uploader", False):
            st.session_state["document_uploader"] = None
        uploaded_file = st.file_uploader(
            "上传 Markdown 或 TXT 文档",
            type=["md", "txt"],
            help="单个文件大小上限由后端配置，默认 5 MB。",
            key="document_uploader",
            disabled=busy,
        )
        if uploaded_file is not None and st.button("上传并建立索引", width="stretch", disabled=busy):
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

    st.markdown("#### 资料库")
    document_filter = st.text_input("筛选文档", placeholder="按主题或文件名筛选", label_visibility="collapsed",
                                    disabled=busy, key="document_filter")
    if st.button("刷新列表", icon=":material/refresh:", type="tertiary", width="stretch", disabled=busy):
        refresh_documents()
    if not st.session_state.documents:
        refresh_documents()
    if not st.session_state.documents:
        st.caption("知识库暂时没有文档。")
    st.caption(f"{len(st.session_state.documents)} 份文档 · 点击查看原文")
    document_titles = {
        "agent_intro.md": "Agent 与核心概念",
        "rag_intro.md": "RAG 基础入门",
        "rag_build_intro.md": "RAG 搭建指南",
        "agentic_rag_build_intro.md": "Agentic RAG 编排",
        "retrieval_quality_guide.md": "检索质量与评估",
        "project_operations_guide.md": "项目使用与排障",
    }
    visible_documents = [d for d in st.session_state.documents
                         if document_filter.casefold() in (d["filename"] + " " + document_titles.get(d["filename"], "")).casefold()]
    if document_filter and not visible_documents:
        st.caption("没有匹配的文档，试试其他关键词")
    for document in visible_documents:
        document_id = document["document_id"]
        with st.container(key=f"document-card-{document_id}" + ("-active" if st.session_state.selected_document_id == document_id else "")):
            if st.button(
                document_titles.get(document["filename"], document["filename"]),
                icon=":material/description:",
                type="tertiary",
                wrap=True,
                key=f"open-{document_id}",
                disabled=busy,
                width="stretch",
            ):
                st.session_state.selected_document_id = (
                    None if st.session_state.selected_document_id == document_id else document_id
                )
            st.caption(f"{document['chunk_count']} 个片段 · {document['origin']}")

    st.space("small")
    if st.button("清空对话", icon=":material/chat_bubble_outline:", width="stretch", disabled=busy):
        _, error = api_request("DELETE", f"/sessions/{st.session_state.session_id}")
        if error:
            st.error(error)
        else:
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.success("会话已清空")
            st.rerun()

document_count = len(st.session_state.documents)
chunk_count = sum(d.get("chunk_count", 0) for d in st.session_state.documents)
mode_name = "Agent 检索" if agent_mode else "经典 RAG"
st.html(f"""
<header class="workspace-header">
  <div><span class="workspace-breadcrumb">本地知识库</span><div class="workspace-wordmark">知识工作台</div></div>
  <div class="workspace-context"><span class="mode-indicator"></span>{mode_name}<span class="header-separator">/</span>{document_count} 份资料</div>
</header>
""")

conversation_pane = st.container(height=600, autoscroll=False, border=False, key="conversation-pane")
with conversation_pane:
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
                with st.container(border=True, key="document-preview"):
                    st.subheader(f"文档预览：{preview['filename']}")
                    st.caption("点击左侧已选中的文档名可收起预览。")
                    language = "markdown" if preview["file_type"] == "md" else "text"
                    reading, original = st.tabs(["阅读视图", "原文"])
                    with reading:
                        if preview["file_type"] == "md":
                            st.markdown(preview["content"])
                        else:
                            st.text(preview["content"])
                    with original:
                        st.code(preview["content"], language=language, wrap_lines=True)

    if not st.session_state.messages and not st.session_state.pending_question:
        st.html(f"""
        <section class="research-intro">
          <div class="intro-copy">
            <p class="intro-eyebrow"><span class="eyebrow-line"></span>基于你的资料，而非凭空回答</p>
            <h1>让每一次回答，<br><em>都能找到出处。</em></h1>
            <p class="intro-description">从一个问题出发，连接文档里的知识。<br>检索过程可查看，引用依据可追溯。</p>
            <div class="library-summary"><span><b>{document_count:02}</b> 知识文档</span><span><b>{chunk_count}</b> 可检索片段</span></div>
          </div>
          <aside class="retrieval-route" aria-label="系统问答路径示意">
            <div class="route-heading">一条答案的来路<span>工作原理</span></div>
            <div class="route-step"><span class="route-number">01</span><div><strong>找到相关资料</strong><p>标题匹配 + 语义检索</p></div><span class="route-symbol">⌕</span></div>
            <div class="route-step"><span class="route-number">02</span><div><strong>整理知识依据</strong><p>片段去重 · 上下文组织</p></div><span class="route-symbol">▤</span></div>
            <div class="route-step"><span class="route-number">03</span><div><strong>形成可核对的回答</strong><p>DeepSeek 生成 · 展示来源</p></div><span class="route-symbol">↗</span></div>
            <div class="route-footnote">答案下方可查看引用片段与工具过程</div>
          </aside>
        </section>
        <div class="question-section-heading"><h2>从一个问题开始</h2><span>选择主题，直接提问 ↗</span></div>
        """)
        suggested_questions = [
            "RAG 是什么？它的基本流程有哪些？",
            "混合检索是什么？关键词检索和向量检索如何互补？",
            "从文档上传到向量入库，RAG 搭建要经过哪些步骤？",
            "Agentic RAG 与传统 RAG 有什么区别？",
        ]
        prompt_notes = ["理解概念与核心链路", "比较两种检索方式", "梳理从文档到答案的完整流程", "了解工具调用与执行循环"]
        for row_start in range(0, len(suggested_questions), 2):
            columns = st.columns(2)
            for offset, column in enumerate(columns):
                question_index = row_start + offset
                if question_index >= len(suggested_questions):
                    continue
                prompt_text = suggested_questions[question_index]
                with column, st.container(key=f"prompt-card-{question_index}"):
                    st.caption(f"0{question_index + 1} / {prompt_notes[question_index]}")
                    if st.button(
                        prompt_text,
                        key=f"suggested-question-{question_index}",
                        wrap=True,
                        disabled=busy,
                        icon=":material/arrow_outward:",
                        icon_position="right",
                        width="stretch",
                    ):
                        st.session_state.pending_question = prompt_text
                        st.rerun()

    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar=":material/person:" if message["role"] == "user" else ":material/hub:"):
            st.caption("你" if message["role"] == "user" else ("Agent" if message.get("mode", "agent") == "agent" else "RAG"))
            st.markdown(message["content"])
            if message.get("error"):
                st.error(message["error"])
            if message.get("warning"):
                st.warning(message["warning"])
            with st.container(horizontal=True, key=f"answer-actions-{message_index}"):
                render_sources(
                    message.get("sources", []),
                    message.get("retrieved_chunks", []),
                    key=f"sources-{message_index}",
                )
                if message.get("mode", "agent") == "agent":
                    render_tool_events(message.get("tool_events", []), key=f"tools-{message_index}")

    # Only the latest successful answer offers actions, avoiding stale duplicate rows.
    if not busy and not st.session_state.pending_question and st.session_state.messages:
        latest = st.session_state.messages[-1]
        if latest["role"] == "assistant" and not latest.get("error"):
            asked = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
            suggestions = suggest_followups(
                asked[-1] if asked else "", latest["content"],
                latest.get("retrieved_chunks", []), st.session_state.documents, asked,
            )
            if suggestions:
                with st.container(key="followup-suggestions"):
                    st.caption("继续探索")
                    with st.container(horizontal=True, gap="small"):
                        for index, prompt in enumerate(suggestions):
                            if st.button(prompt, key=f"followup-{len(st.session_state.messages)}-{index}",
                                         icon=":material/subdirectory_arrow_right:", wrap=True):
                                st.session_state.pending_question = prompt
                                st.rerun()

question = st.chat_input("在这里输入你的问题...", disabled=busy) or st.session_state.pop("pending_question", None)
if question and not busy:
    st.session_state.pending_request = {"question": question, "mode": "agent" if agent_mode else "rag"}
    st.session_state.is_generating = True
    # Render disabled controls BEFORE starting the blocking stream.
    st.rerun()

if busy and "pending_request" in st.session_state:
    request = st.session_state.pop("pending_request")
    question, mode = request["question"], request["mode"]
    with conversation_pane:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(question)
        with st.chat_message("assistant", avatar=":material/hub:"):
            metadata = {
                "tool_events": [],
                "sources": [],
                "retrieved_chunks": [],
                "error": None,
                "mode": mode,
            }

            status = (
                st.status("Agent 正在分析并调用工具…", expanded=True)
                if mode == "agent" else None
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

            try:
                answer = st.write_stream(
                    stream_chat(
                        st.session_state.session_id,
                        question,
                        mode,
                        metadata,
                        on_tool_event=log_tool_event,
                    )
                )
            except Exception:
                answer = ""
                metadata["error"] = "回答显示失败，请重新提问。"
            finally:
                st.session_state.is_generating = False
            if metadata["error"]:
                if status is not None:
                    status.update(label="Agent 调用失败", state="error")
                st.error(metadata["error"])
                st.session_state.messages.append({
                    "role": "assistant", "content": answer or "本轮回答未完成。",
                    "error": metadata["error"], "mode": mode,
                })
            else:
                if status is not None:
                    status.update(label="Agent 调用完成", state="complete", expanded=True)
                render_sources(metadata["sources"], metadata["retrieved_chunks"], key="sources-live")
                if metadata.get("warning"):
                    st.warning(metadata["warning"])
                mode_label = "Agent" if metadata["mode"] == "agent" else "传统 RAG"
                st.caption(f"回答模式：{mode_label}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": metadata["sources"],
                    "retrieved_chunks": metadata["retrieved_chunks"],
                    "tool_events": metadata["tool_events"],
                    "mode": metadata["mode"],
                    "warning": metadata.get("warning"),
                })

    st.rerun()
