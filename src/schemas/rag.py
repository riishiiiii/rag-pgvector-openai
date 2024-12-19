from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from enum import Enum
from dataclasses import dataclass



class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence_score: float
    context_id: Optional[str] = None


class Status(Enum):
    INGESTING = "Ingesting"
    VECTORING = "Vectorizing"
    COMPLETED = "Completed"
    FAILED = "Failed"


class DocumentTaskResponse(BaseModel):
    task_id: str
    status: Status
    progress: int
    context_id: Optional[str] = None


class RAGConfig(BaseModel):
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0
    max_tokens: int = 500
    top_k: int = 4



@dataclass
class DocumentUploadConfig:
    DEFAULT_MODEL = "gpt-3.5-turbo"
    DEFAULT_TEMP = 0.0
    DEFAULT_MAX_TOKENS = 500
    DEFAULT_TOP_K = 4
