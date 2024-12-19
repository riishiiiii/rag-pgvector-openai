# RAG API Service

This project provides APIs for managing document uploads and performing Retrieval-Augmented Generation (RAG) operations. The backend is built using FastAPI, and the services are containerized using Docker.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Services Overview](#services-overview)
- [API Endpoints](#api-endpoints)
  - [Upload Document](#upload-document)
  - [Get Document Status](#get-document-status)
  - [Ask a Question](#ask-a-question)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)

---

## Prerequisites

- Docker and Docker Compose installed
- Python 3.8 or higher

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/riishiiiii/rag-pgvector-openai
   cd rag-pgvector-openai
   ```

2. Create a `.env` file and configure environment variables (refer to [Environment Variables](#environment-variables)).

3. Start the services using Docker Compose:
   ```bash
   docker-compose up --build
   ```

4. The backend service will be available at [http://localhost:8000](http://localhost:8000).

## Services Overview

### Backend
A FastAPI application handling document uploads, status tracking, and question-answering functionality.

### pg_vector
A PostgreSQL database with the `pgvector` extension for vector similarity search.

### Redis
Used as a caching layer for storing task statuses and other transient data.

## API Endpoints

### 1. **Upload Document**
   **Endpoint:** `/documents/upload`

   **Method:** `POST`

   **Description:** Uploads a document (PDF, TXT, or Markdown) for processing within a specific context.

   **Request Headers:**
   - `x-context-id` (required): A unique context identifier.

   **Request Body:**
   - `file` (UploadFile): The document file to upload.

   **Response:**
   ```json
   {
     "task_id": "<unique-task-id>",
     "status": "INGESTING",
     "progress": 0,
     "context_id": "<context-id>"
   }
   ```

### 2. **Get Document Status**
   **Endpoint:** `/documents/status/{task_id}`

   **Method:** `GET`

   **Description:** Retrieves the current status of a document processing task.

   **Response:**
   ```json
   {
     "task_id": "<task-id>",
     "status": "<status>",
     "progress": <progress>,
     "context_id": "<context-id>"
   }
   ```

### 3. **Ask a Question**
   **Endpoint:** `/rag/question`

   **Method:** `POST`

   **Description:** Queries the RAG system for an answer based on uploaded documents.

   **Request Headers:**
   - `x-context-id` (required): A unique context identifier.

   **Request Body:**
   ```json
   {
     "question": "<your-question>"
   }
   ```

   **Query Parameters:**
   - `model_name` (optional, default: "gpt-3.5-turbo"): The name of the language model to use.
   - `temperature` (optional, default: 0.7): The randomness level for response generation (0-1).
   - `max_tokens` (optional, default: 256): The maximum number of tokens in the response.
   - `top_k` (optional, default: 5): The number of relevant documents to retrieve (1-20).

   **Response:**
   ```json
   {
     "answer": "<model-generated-answer>",
     "sources": [
       "<relevant-source-document>",
       "<relevant-source-document>"
     ],
     "confidence_score": <confidence-score>,
     "context_id": "<context-id>"
   }
   ```

## Environment Variables

Create a `.env` file with the following variables:

```env
REDIS_HOST="redis"
REDIS_PORT="6379"
OPENAI_API_KEY="<your-api-key>"
```

## Project Structure

```plaintext
.
├── app.py
├── config.py
├── docker-compose.yml
├── dockerfile
├── Readme.MD
├── requirements.txt
├── src
│   ├── database
│   │   ├── database.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── pgvector_rag
│   │   ├── __init__.py
│   │   ├── pgrag.py
│   │   └── vector_store.py
│   ├── routes
│   │   ├── __init__.py
│   │   └── rag.py
│   ├── schemas
│   │   ├── __init__.py
│   │   └── rag.py
│   └── service
│       ├── docling_service.py
│       ├── __init__.py
│       ├── rag_service.py
│       └── redis_service.py
└── temp
```

---

## Notes

- Make sure the `OPENAI_API_KEY` is set in the `.env` file to enable RAG functionality.

