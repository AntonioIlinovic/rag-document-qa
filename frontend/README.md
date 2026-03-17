# RAG Document QA - Streamlit Frontend

A modern Streamlit interface for the RAG Document QA system.

## Features

- **Native Streamlit Chat Components** - Clean, modern chat interface
- **Sequential User Flow** - Step-by-step guidance: Session → Documents → Chat
- **Session Management** - Create or join sessions for document isolation
- **Document Upload** - Support for PDF, images, and text files (only after session creation)
- **Detailed Answer Display** - Always shows comprehensive answer details:
  - Processing metrics (time, sources used)
  - Source chunks with relevance scores
  - Model information and confidence levels
- **Real-time Processing** - Live feedback during document processing and Q&A

## Usage

### Development (Local)

1. Install dependencies:
```bash
cd frontend
pip install -e .
```

2. Run the app:
```bash
streamlit run app.py --server.port 8501
```

3. Access at `http://localhost:8501`

### Docker

1. Build and run with docker-compose:
```bash
docker compose up --build
```

2. Access at `http://localhost:8501`

## Interface Overview

### Sidebar
- **Backend Status** - Connection indicator
- **Session Management** - Create/join sessions (required first step)
- **Document Upload** - Multi-file upload with progress (only available after session creation)
- **Document List** - View uploaded documents

### Main Chat Interface
- **Sequential Workflow** - Clear step-by-step process:
  1. Create/join session
  2. Upload documents
  3. Start chatting
- **Detailed Answers** - All answers automatically show:
  - Processing metrics (time, sources used)
  - Source chunks with relevance scores
  - Model information and confidence levels

### Answer Details
Every answer shows comprehensive details:
- **📊 Query Processing** - Question, sources used, processing time
- **📚 Source Chunks Used** - Each chunk with:
  - Relevance score and confidence indicator (🟢🟡🔴)
  - Source document information
  - Expanded by default for easy viewing
- **🤖 Model Response** - QA engine used and answer confidence

## Configuration

Environment variables:
- `BACKEND_URL` - Backend API URL (default: `http://backend:8000`)
- `OPENAI_API_KEY` - OpenAI API key (if using cloud QA)
- `QA_ENGINE` - QA engine type (`cloud` or `local`, default: `local`)

## Architecture

- **BackendClient** - HTTP client for API communication
- **Session State** - Manages chat history, documents, and session ID
- **UI Components** - Modular sidebar and chat interface
- **Error Handling** - Graceful error display and recovery

## Dependencies

- `streamlit>=1.39.0` - Main UI framework
- `httpx>=0.28.1` - Async HTTP client
- `pandas>=2.0.0` - Data display utilities
