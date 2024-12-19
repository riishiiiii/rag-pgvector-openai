from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Header, Query
import shutil
import aiofiles
import os
import uuid
from typing import Optional

from ..schemas.rag import (
    QuestionRequest,
    QuestionResponse,
    DocumentTaskResponse,
    Status,
    DocumentUploadConfig,
    RAGConfig,
)
from ..service.rag_service import RagService

router = APIRouter()


async def get_context_id(x_context_id: Optional[str] = Header(None)) -> Optional[str]:
    """
    Dependency to extract context ID from headers.
    You can modify this to get context from wherever makes sense for your application
    (e.g., JWT tokens, query parameters, etc.)
    """
    return x_context_id


async def get_rag_service(context_id: Optional[str] = Depends(get_context_id)) -> RagService:
    return RagService(context_id=context_id)


@router.post(
    "/documents/upload",
    # dependencies=[Depends(ApiKeyMiddleware()), Depends(RateLimiter(times=5, seconds=60))],
    response_model=DocumentTaskResponse,
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to upload (PDF, TXT, or MD)"),
    context_id: str = Depends(get_context_id),
) -> DocumentTaskResponse:
    """
    Upload a PDF document for RAG processing within a specific context.
    """
    if not context_id:
        raise HTTPException(status_code=400, detail="Context ID is required for document upload")

    unique_filename = f"{context_id}_{uuid.uuid4()}_{file.filename}"
    temp_file_path = os.path.join("temp", unique_filename)
    os.makedirs("temp", exist_ok=True)

    try:
        async with aiofiles.open(temp_file_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        task_id = str(uuid.uuid4())
        service = RagService(context_id=context_id)
        # Add context_id to the background task
        background_tasks.add_task(service.process_document, temp_file_path, task_id, context_id)

        # Initialize task status in cache with context information
        await service.cache_service.set(
            task_id,
            {
                "status": Status.INGESTING.value,
                "progress": 0,
                "context_id": context_id,
            },
        )

        return DocumentTaskResponse(task_id=task_id, status=Status.INGESTING.value, progress=0, context_id=context_id)

    except Exception as e:
        # Clean up the file in case of error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/status/{task_id}", response_model=DocumentTaskResponse)
async def get_document_status(task_id: str, service: RagService = Depends(get_rag_service)) -> DocumentTaskResponse:
    """
    Get the status of a document processing task within a context.
    """
    try:
        task_data = await service.get_processing_status(task_id)
        if not task_data:
            raise HTTPException(status_code=404, detail="Task not found")

        return DocumentTaskResponse(
            task_id=task_id,
            status=task_data["status"],
            progress=task_data.get("progress", 0),
            context_id=task_data.get("context_id"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/question")
async def ask_question(
    request: QuestionRequest,
    model_name: str = Query(default=DocumentUploadConfig.DEFAULT_MODEL),
    temperature: float = Query(default=DocumentUploadConfig.DEFAULT_TEMP, ge=0.0, le=1.0),
    max_tokens: int = Query(default=DocumentUploadConfig.DEFAULT_MAX_TOKENS),
    top_k: int = Query(default=DocumentUploadConfig.DEFAULT_TOP_K, ge=1, le=20),
    context_id: str = Depends(get_context_id),
) -> QuestionResponse:
    """
    Ask a question using the RAG system within a specific context.

    Args: \n
        model_name: Name of the LLM model (e.g., gpt-3.5-turbo, gpt-4) \n
        temperature: Model temperature for response generation (0-1) \n
        max_tokens: Maximum tokens in model response \n
        top_k: Number of relevant documents to retrieve \n
        context_id: Context ID for isolating the document \n
    """
    try:
        rag_config = RAGConfig(model_name=model_name, temperature=temperature, max_tokens=max_tokens, top_k=top_k)
        service = RagService(context_id=context_id, config=rag_config)
        return await service.get_answer(question=request.question, context_id=context_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


