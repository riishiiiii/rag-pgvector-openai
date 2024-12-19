from ..service.docling_service import DoclingFileLoader
from ..service.redis_service import RedisCacheService
from ..pgvector_rag.vector_store import VectorStore
from ..pgvector_rag.pgrag import PgRAGSystem
from ..schemas.rag import RAGConfig, Status, QuestionResponse

import asyncio
from typing import Optional, Dict, Any, List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RagService:

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        context_id: Optional[str] = None,
    ):
        self.vector_store = VectorStore(context_id)
        self.config = config or RAGConfig()
        print(self.config)
        self.rag_system = PgRAGSystem(context_id, config=self.config)
        self.cache_service = RedisCacheService()

    async def process_document(self, file_path: str, task_id: str, context_id: Optional[str] = None) -> int:
        """
        Process a PDF document and store it in the vector store with context isolation.

        Args:
            file_path: Path to the PDF file
            task_id: ID for tracking processing status
            context_id: ID for isolating the document in its own context
        Returns:
            document ID
        """
        try:
            await self._create_status(task_id, Status.INGESTING, 0)

            # Extract text from PDF
            texts = await asyncio.to_thread(self._extract_text_from_pdf, file_path)

            # Split text into chunks
            chunks = await asyncio.to_thread(self._chunk_text, texts)
            await self._create_status(task_id, Status.VECTORING, 50)

            # Generate document ID
            doc_id = hash(f"{context_id}:{file_path}" if context_id else file_path)

            # Create metadata with context information
            metadatas = await self._generate_metadata(doc_id, file_path, chunks, context_id)

            # Add to vector store with context isolation
            await self.vector_store.add_documents(texts=chunks, metadatas=metadatas)

            await self._create_status(task_id, Status.COMPLETED, 100)
            return doc_id
        

        except Exception as e:
            await self._create_status(task_id, Status.FAILED, error=str(e))
            raise Exception(f"Error processing document: {str(e)}")

        finally:
            # Clean up the file
            import os
            if os.path.exists(file_path):
                os.remove(file_path)

    async def _generate_metadata(
        self, doc_id: str, file_path: str, chunks: List[str], context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate metadata for a document with context information."""
        return [
            {
                "source": file_path,
                "chunk": i,
                "total_chunks": len(chunks),
                "doc_id": doc_id,
                "context_id": context_id,
            }
            for i in range(len(chunks))
        ]

    async def _create_status(self, task_id: str, status: Status, progress: int = 0, error: str = None) -> None:
        """Create a status entry for a task in the cache."""
        status_data = {"status": status.value, "progress": progress, **({"error": error} if error else {})}
        await self.cache_service.set(task_id, status_data)

    async def get_answer(
        self,
        question: str,
        context_id: Optional[str] = None,
    ) -> QuestionResponse:
        # """
        # Get an answer for a question using the RAG system within a specific context.
# 
        # Args:
            # question: The question to answer
            # context_id: Optional context to restrict the search
        # """
        # try:
            response = await self.rag_system.query(question)
            sources = [doc.metadata.get("source") for doc in response["source_documents"]]
            confidence_score = await asyncio.to_thread(self._calculate_confidence_score, response, context_id)

            return QuestionResponse(
                answer=response["answer"], sources=sources, confidence_score=confidence_score, context_id=context_id
            )
        # except Exception as e:
            # raise Exception(f"Error getting answer: {str(e)}")

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        loader = DoclingFileLoader(file_path=file_path)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        return " ".join([split.page_content for split in splits])

    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks of approximately equal size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > chunk_size:
                if current_chunk:  # Avoid empty chunks
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _calculate_confidence_score(self, response: Dict[str, Any], context_id: Optional[str] = None) -> float:
        """
        Calculate a confidence score based on the response and context.
        """
        sources = len(response.get("source_documents", []))
        if sources == 0:
            return 0.0

        # Calculate relevance score based on source documents
        source_scores = []
        for doc in response["source_documents"]:
            # Verify context match if context_id is provided
            if context_id and doc.metadata.get("context_id") != context_id:
                continue

            # Add your relevance scoring logic here
            content_length = len(doc.page_content)
            source_scores.append(min(content_length / 500, 1.0))

        if not source_scores:
            return 0.0

        avg_source_score = sum(source_scores) / len(source_scores)

        # Calculate answer quality score
        answer_length = len(response.get("answer", ""))
        answer_score = min(answer_length / 200, 1.0)

        # Weighted combination
        final_score = avg_source_score * 0.7 + answer_score * 0.3
        return round(final_score, 2)


    async def get_processing_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the current status of a batch processing task.

        Args:
            task_id: The ID of the batch processing task

        Returns:
            dict: The current status of the processing task

        Raises:
            Exception: If the task is not found
        """
        status = await self.cache_service.get(task_id)
        if not status:
            raise Exception(f"No status found for task ID: {task_id}")
        return status
