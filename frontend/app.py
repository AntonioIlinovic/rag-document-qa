import os
import time
import traceback
from typing import Dict, List, Optional, Tuple
import uuid
import yaml

import httpx
import pandas as pd
import streamlit as st

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL")

# Load shared configuration
def load_shared_config():
    """Load file upload configuration from shared_config.yaml."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "shared_config.yaml")
        with open(config_path, 'r') as f:
            shared_config = yaml.safe_load(f)
        return shared_config.get('file_upload', {})
    except (FileNotFoundError, yaml.YAMLError, KeyError):
        # Fallback to defaults if shared config is not available
        return {
            'supported_extensions': ["pdf", "png", "jpg", "jpeg", "tiff", "txt", "md"]
        }

# Load configuration at startup
shared_config = load_shared_config()
SUPPORTED_EXTENSIONS = shared_config.get('supported_extensions', [])

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0

# API client functions
class BackendClient:
    """Client for interacting with the backend API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.timeout = 30.0
    
    def health_check(self) -> bool:
        """Check if backend is healthy."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/health/")
                return response.status_code == 200
        except Exception:
            return False
    
    def upload_documents(self, files: List[bytes], filenames: List[str], session_id: Optional[str] = None) -> Dict:
        """Upload documents to the backend."""
        with httpx.Client(timeout=self.timeout) as client:
            files_data = [
                ("files", (filename, file_content))
                for filename, file_content in zip(filenames, files)
            ]
            
            data = {}
            if session_id:
                data["session_id"] = session_id
            
            response = client.post(
                f"{self.base_url}/upload/",
                files=files_data,
                data=data,
                follow_redirects=True
            )
            response.raise_for_status()
            return response.json()
    
    def ask_question(self, session_id: str, question: str) -> Dict:
        """Ask a question about uploaded documents."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/ask/",
                json={"session_id": session_id, "question": question},
                follow_redirects=True
            )
            response.raise_for_status()
            return response.json()

# UI Components
def render_answer_details(answer_data: dict):
    """Renders the collapsible details block for an assistant message."""
    if not answer_data:
        return
    
    sources = answer_data.get("sources", [])
    qa_model = answer_data.get("qa_model", "Unknown")
    embedding_model = answer_data.get("embedding_model", "Unknown")
    processing_time = answer_data.get("processing_time")
    
    # Single expander for details
    with st.expander("How was this answer retrieved?", expanded=False):
        # Processing info
        if processing_time is not None:
            st.caption(f"Processing time: {processing_time:.1f}s")
        
        # Models used section
        st.caption(f"Embedding model: {embedding_model}")
        st.caption(f"QA model: {qa_model}")
        
        # Per-chunk nested expanders
        for i, source in enumerate(sources, 1):
            score = source.get("score", 0.0)
            chunk = source.get("chunk", "")
            metadata = source.get("metadata", {})
            
            # Color indicator logic
            if score >= 0.8:
                color_indicator = "🟢"
            elif score >= 0.6:
                color_indicator = "🟡"
            else:
                color_indicator = "🔴"
            
            # Extract filename and page from metadata
            filename = metadata.get('filename', 'Unknown')
            page = metadata.get('page')
            page_text = f"p.{page}" if page else ""
            
            expander_label = f"Chunk {i} — {filename}"
            if page_text:
                expander_label += f"  ({page_text})"
            expander_label += f"  ·  similarity {score:.3f}  {color_indicator}"
            
            with st.expander(expander_label, expanded=False):
                st.markdown(chunk)
                
                # Source info
                st.caption(f"File: {filename}")
                if page:
                    st.caption(f"Page: {page}")
                st.caption(f"Cosine similarity to your question: {score:.3f}")

def render_sidebar(client: BackendClient):
    """Render the sidebar with document upload and session management."""
    st.sidebar.title("📚 RAG Document QA")
    
    # Backend status
    if client.health_check():
        st.sidebar.success("🟢 Backend Connected")
    else:
        st.sidebar.error("🔴 Backend Disconnected")
        st.sidebar.info("Please ensure the backend is running")
        return
    
    # Session management
    st.sidebar.subheader("Session Management")
    
    # Join existing session
    if st.session_state.session_id:
        st.sidebar.info(f"Current Session: {st.session_state.session_id}")
        if st.sidebar.button("Copy Session ID"):
            st.experimental_copy(st.session_state.session_id)
    else:
        st.sidebar.info("Upload documents to create a new session, or enter an existing session ID below:")
        session_input = st.sidebar.text_input("Or enter existing session ID:")
        if session_input and st.sidebar.button("Join Session"):
            st.session_state.session_id = session_input.strip()
            st.session_state.documents = []
            st.session_state.chat_messages = []
            st.rerun()
    
    # Document upload (always visible)
    st.sidebar.subheader("Document Upload")
    
    uploaded_files = st.sidebar.file_uploader(
        f"Upload documents ({', '.join(SUPPORTED_EXTENSIONS).upper()})",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        help="Upload one or more documents to analyze",
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )
    
    if uploaded_files and st.sidebar.button("Upload Documents", type="primary"):
        with st.sidebar.spinner("Processing documents..."):
            try:
                files = [f.getvalue() for f in uploaded_files]
                filenames = [f.name for f in uploaded_files]
                
                result = client.upload_documents(files, filenames, st.session_state.session_id)
                
                # Update session ID if it was created by the backend
                if not st.session_state.session_id:
                    st.session_state.session_id = result.get("session_id")
                
                # Update documents list
                for doc in result.get("documents", []):
                    st.session_state.documents.append({
                        "filename": doc["filename"],
                        "chunks": doc["chunks"],
                        "status": doc["status"]
                    })
                
                # Clear the file uploader by incrementing the key
                st.session_state.file_uploader_key += 1
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"❌ Upload failed: {str(e)}")
    
    # Display uploaded documents
    if st.session_state.documents:
        st.sidebar.subheader("Uploaded Documents")
        
        doc_df = pd.DataFrame(st.session_state.documents)
        st.sidebar.dataframe(doc_df, width='stretch')
        
        if st.sidebar.button("Clear All Documents"):
            st.session_state.documents = []
            st.session_state.chat_messages = []
            st.rerun()


def render_chat_interface(client: BackendClient):
    """Render the main chat interface."""
    # Check if session exists
    if not st.session_state.session_id:
        st.info("Please create or join a session in the sidebar to start chatting")
        return
    
    # Check if documents are uploaded
    if not st.session_state.documents:
        st.info("Please upload some documents in the sidebar to start asking questions")
        return
    
    # Display welcome message always
    st.markdown("""
    ## 👋 Welcome to RAG Document QA!
    
    Start by asking a question about your uploaded documents. I'll search through them and provide answers based on the content.
    """)
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                # Assistant message - just show the answer text
                st.write(message["content"])
                
                # Add the collapsible details section
                if "details" in message and message["details"]:
                    render_answer_details(message["details"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.processing:
            # Add user message
            st.session_state.chat_messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Set processing state
            st.session_state.processing = True
            
            # Process the question
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching documents..."):
                    try:
                        start_time = time.time()
                        result = client.ask_question(st.session_state.session_id, prompt)
                        processing_time = time.time() - start_time
                        
                        answer = result.get("answer", "Sorry, I couldn't generate an answer.")
                        sources = result.get("sources", [])
                        
                        # Add processing time to result for display
                        result["processing_time"] = processing_time
                        
                        # Display answer text directly (no header)
                        st.write(answer)
                        
                        # Add assistant message to chat history
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": answer,
                            "details": result,
                            "question": prompt
                        })
                        
                        # Show detailed view using the helper function
                        if sources:
                            render_answer_details(result)
                        
                    except httpx.HTTPStatusError as e:
                        error_msg = "❌ Sorry, I encountered an error processing your question."
                        st.write(error_msg)
                        
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "details": None,
                            "question": prompt
                        })
                        
                        if e.response.status_code == 404:
                            st.error("Session not found. Please create a new session.")
                        elif e.response.status_code == 400:
                            st.error("No documents found in session. Please upload documents first.")
                        else:
                            st.error(f"Backend error: {e.response.text}")
                    
                    except Exception as e:
                        error_msg = "❌ Sorry, I encountered an unexpected error."
                        st.write(error_msg)
                        
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "details": None,
                            "question": prompt
                        })
                        
                        st.error(f"Error: {str(e)}")
                        if st.checkbox("Show traceback"):
                            st.code(traceback.format_exc())
                    
                    finally:
                        st.session_state.processing = False
                        st.rerun()
        else:
            st.warning("Please wait for the current question to be processed...")

def main():
    """Main application entry point."""
    init_session_state()
    
    # Check if BACKEND_URL is set
    if not BACKEND_URL:
        st.error("❌ BACKEND_URL environment variable is not set!")
        return
    
    # Initialize backend client
    client = BackendClient(BACKEND_URL)
    
    # Render layout
    render_sidebar(client)
    render_chat_interface(client)

if __name__ == "__main__":
    main()
