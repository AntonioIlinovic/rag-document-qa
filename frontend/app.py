import os
import time
import traceback
from typing import Dict, List, Optional
import yaml

import httpx
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import NER highlighting component
from components.ner_highlighting import render_answer_with_entities
from components.config_section import render_config_section, get_qa_engine, is_ner_enabled

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL")


def load_shared_config():
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "shared_config.yaml")
        with open(config_path, 'r') as f:
            shared_config = yaml.safe_load(f)
        return shared_config.get('file_upload', {})
    except (FileNotFoundError, yaml.YAMLError, KeyError):
        return {'supported_extensions': ["pdf", "png", "jpg", "jpeg", "tiff", "txt", "md"]}


shared_config = load_shared_config()
SUPPORTED_EXTENSIONS = shared_config.get('supported_extensions', [])


# ── Session state ────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "session_id": None,
        "chat_messages": [],
        "documents": [],
        "processing": False,
        "file_uploader_key": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Backend client ───────────────────────────────────────────────────────────

class BackendClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.timeout = 300.0

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/health/")
                return response.status_code == 200
        except Exception:
            return False

    def upload_documents(self, files: List[bytes], filenames: List[str], session_id: Optional[str] = None) -> Dict:
        with httpx.Client(timeout=self.timeout) as client:
            files_data = [("files", (fn, fc)) for fn, fc in zip(filenames, files)]
            data = {"session_id": session_id} if session_id else {}
            response = client.post(f"{self.base_url}/upload/", files=files_data, data=data, follow_redirects=True)
            response.raise_for_status()
            return response.json()

    def ask_question(self, session_id: str, question: str) -> Dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/ask/",
                json={
                    "session_id": session_id, 
                    "question": question,
                    "qa_engine": get_qa_engine(),
                    "ner_enabled": is_ner_enabled()
                },
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.json()

    def load_sessions(self) -> List[Dict]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/sessions/")
                response.raise_for_status()
                return response.json().get("sessions", [])
        except Exception:
            return []

    def load_session_details(self, session_id: str) -> Dict:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/sessions/{session_id}")
                response.raise_for_status()
                return response.json()
        except Exception:
            return {}

    def save_chat_message(self, session_id: str, role: str, content: str, details: Dict = None) -> bool:
        """Save a chat message to session history."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                data = {
                    "session_id": session_id,
                    "role": role,
                    "content": content
                }
                if details is not None:
                    data["details"] = details
                
                response = client.post(f"{self.base_url}/chat/message", json=data, follow_redirects=True)
                response.raise_for_status()
                return True
        except Exception:
            return False

    def load_chat_history(self, session_id: str) -> List[Dict]:
        """Load chat history for a session."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/chat/history/{session_id}")
                response.raise_for_status()
                data = response.json()
                
                # Convert API response to frontend format
                messages = []
                for msg in data.get("messages", []):
                    message = {
                        "role": msg["role"],
                        "content": msg["content"],
                        "details": msg.get("details")
                    }
                    messages.append(message)
                
                return messages
        except Exception:
            return []


@st.cache_data(ttl=30)
def get_sessions_cached(_client: BackendClient) -> List[Dict]:
    return _client.load_sessions()


# ── UI helpers ───────────────────────────────────────────────────────────────

def switch_session(session_id: str, client: BackendClient):
    """Load a past session into state."""
    st.session_state.session_id = session_id
    st.session_state.processing = False
    
    # Load documents
    details = client.load_session_details(session_id)
    st.session_state.documents = [
        {"filename": d["filename"], "chunks": d["chunks"], "status": d["status"]}
        for d in details.get("documents", [])
    ]
    
    # Load chat history
    st.session_state.chat_messages = client.load_chat_history(session_id)


def new_session():
    """Clear state to start a fresh session."""
    st.session_state.session_id = None
    st.session_state.chat_messages = []
    st.session_state.documents = []
    st.session_state.processing = False
    st.session_state.file_uploader_key += 1


def render_answer_details(answer_data: dict):
    if not answer_data:
        return

    sources = answer_data.get("sources", [])
    qa_model = answer_data.get("qa_model", "Unknown")
    embedding_model = answer_data.get("embedding_model", "Unknown")
    processing_time = answer_data.get("processing_time")

    with st.expander("How was this answer retrieved?", expanded=False):
        if processing_time is not None:
            st.caption(f"Processing time: {processing_time:.1f}s")
        st.caption(f"Embedding model: {embedding_model}")
        st.caption(f"QA model: {qa_model}")

        for i, source in enumerate(sources, 1):
            score = source.get("score", 0.0)
            chunk = source.get("chunk", "")
            metadata = source.get("metadata", {})

            color = "🟢" if score >= 0.8 else ("🟡" if score >= 0.6 else "🔴")
            filename = metadata.get('filename', 'Unknown')
            page = metadata.get('page')

            label = f"Chunk {i} — {filename}"
            if page:
                label += f" (p.{page})"
            label += f"  ·  similarity {score:.3f}  {color}"

            with st.expander(label, expanded=False):
                st.markdown(chunk)
                st.caption(f"File: {filename}")
                if page:
                    st.caption(f"Page: {page}")
                st.caption(f"Cosine similarity: {score:.3f}")


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(client: BackendClient):
    st.sidebar.title("RAG Document QA")

    if not client.health_check():
        st.sidebar.error("🔴 Backend Disconnected")
        st.sidebar.info("Please ensure the backend is running")
        return

    st.sidebar.success("🟢 Backend Connected")
    st.sidebar.divider()

    # ── Past sessions selectbox ──────────────────────────────────────────────
    sessions = get_sessions_cached(client)

    if sessions:
        st.sidebar.subheader("Past Sessions")

        def format_session(s):
            filenames = s["filenames"]
            if len(filenames) <= 2:
                label = ", ".join(filenames)
            else:
                label = f"{filenames[0]}, {filenames[1]} +{len(filenames) - 2} more"
            return f"{label} ({s['document_count']} docs)"

        # Find which session to show selected in the box (None = no match = index 0)
        session_ids = [s["session_id"] for s in sessions]
        active_index = session_ids.index(st.session_state.session_id) if st.session_state.session_id in session_ids else 0

        selected = st.sidebar.selectbox(
            "Load a past session:",
            options=sessions,
            format_func=format_session,
            index=active_index,
            key="session_selectbox",
        )

        if st.sidebar.button("Load Session", use_container_width=True):
            switch_session(selected["session_id"], client)
            st.rerun()

    # ── New session button ───────────────────────────────────────────────────
    st.sidebar.subheader("New Session")

    if st.sidebar.button("＋ New Session", use_container_width=True):
        new_session()
        st.rerun()

    # ── Document upload ──────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Upload Documents")

    # Show which session documents will be added to
    if st.session_state.session_id:
        st.sidebar.caption("Adding to current session")
    else:
        st.sidebar.caption("Will create a new session")

    uploaded_files = st.sidebar.file_uploader(
        f"Supported: {', '.join(SUPPORTED_EXTENSIONS).upper()}",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}",
    )

    if uploaded_files and st.sidebar.button("Upload", type="primary", use_container_width=True):
        with st.sidebar.spinner("Processing..."):
            try:
                result = client.upload_documents(
                    [f.getvalue() for f in uploaded_files],
                    [f.name for f in uploaded_files],
                    st.session_state.session_id,
                )

                st.session_state.session_id = result.get("session_id")

                for doc in result.get("documents", []):
                    st.session_state.documents.append({
                        "filename": doc["filename"],
                        "chunks": doc["chunks"],
                        "status": doc["status"],
                    })

                st.session_state.file_uploader_key += 1
                get_sessions_cached.clear()
                st.rerun()

            except Exception as e:
                st.sidebar.error(f"❌ Upload failed: {str(e)}")

    # ── Uploaded docs list ───────────────────────────────────────────────────
    if st.session_state.documents:
        st.sidebar.markdown(f"**Documents in session ({len(st.session_state.documents)})**")
        for doc in st.session_state.documents:
            st.sidebar.markdown(f"&nbsp;&nbsp;📄 **{doc['filename']}** ({doc['chunks']} chunks)")

    # ── Configuration section ─────────────────────────────────────────────
    st.sidebar.divider()
    render_config_section()


# ── Chat interface ───────────────────────────────────────────────────────────

def render_chat_interface(client: BackendClient):
    if not st.session_state.session_id:
        st.info("Upload documents or load a past session from the sidebar to get started.")
        return

    if not st.session_state.documents:
        st.info("Upload some documents in the sidebar to start asking questions.")
        return

    st.markdown("## 👋 Welcome to RAG Document QA!\n\nAsk a question about your uploaded documents.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("details"):
                # Use NER rendering if available and enabled
                details = message["details"]
                entities = details.get("entities", [])
                
                if entities and is_ner_enabled():
                    render_answer_with_entities(
                        answer=message["content"],
                        entities=entities
                    )
                else:
                    st.write(message["content"])
                
                if details.get("sources"):
                    render_answer_details(details)
            else:
                st.write(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        if st.session_state.processing:
            st.warning("Please wait for the current question to finish...")
            return

        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Save user message to chat history
        client.save_chat_message(st.session_state.session_id, "user", prompt)

        st.session_state.processing = True

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                try:
                    start_time = time.time()
                    result = client.ask_question(st.session_state.session_id, prompt)
                    result["processing_time"] = time.time() - start_time

                    answer = result.get("answer", "Sorry, I couldn't generate an answer.")
                    entities = result.get("entities", [])
                    
                    # Render answer with NER highlighting if enabled
                    if entities and is_ner_enabled():
                        render_answer_with_entities(
                            answer=answer,
                            entities=entities
                        )
                    else:
                        st.write(answer)

                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "details": result,
                    })

                    # Save assistant message to chat history
                    client.save_chat_message(
                        st.session_state.session_id, 
                        "assistant", 
                        answer, 
                        details=result
                    )

                    if result.get("sources"):
                        render_answer_details(result)

                except httpx.HTTPStatusError as e:
                    msg = "❌ Sorry, I encountered an error processing your question."
                    st.write(msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": msg, "details": None})

                    # Save error message to chat history
                    client.save_chat_message(st.session_state.session_id, "assistant", msg)

                    if e.response.status_code == 404:
                        st.error("Session not found. Please create a new session.")
                    elif e.response.status_code == 400:
                        st.error("No documents found in session. Please upload documents first.")
                    else:
                        st.error(f"Backend error: {e.response.text}")

                except Exception as e:
                    msg = "❌ Sorry, I encountered an unexpected error."
                    st.write(msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": msg, "details": None})

                    # Save error message to chat history
                    client.save_chat_message(st.session_state.session_id, "assistant", msg)
                    st.error(f"Error: {str(e)}")
                    if st.checkbox("Show traceback"):
                        st.code(traceback.format_exc())

                finally:
                    st.session_state.processing = False
                    st.rerun()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    init_session_state()

    if not BACKEND_URL:
        st.error("❌ BACKEND_URL environment variable is not set!")
        return

    client = BackendClient(BACKEND_URL)
    render_sidebar(client)
    render_chat_interface(client)


if __name__ == "__main__":
    main()