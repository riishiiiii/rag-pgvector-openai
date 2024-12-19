from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document as LangchainDocument
from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document
from functools import lru_cache
from pydantic import Field
import asyncio
from langchain_postgres.vectorstores import PGVector


from config import Settings
from ..database.database import get_database_url, get_db

@lru_cache()
def get_settings() -> Settings:
    return Settings()


class CustomRetriever(BaseRetriever):
    """Custom retriever that works with our VectorStore."""

    vectorstore: Any
    search_kwargs: dict = Field(default_factory=lambda: {"k": 4})
    context_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    async def _aget_relevant_documents(self, query: str) -> List[LangchainDocument]:
        """Get documents relevant to the query."""
        return await self.vectorstore.similarity_search(query, k=self.search_kwargs.get("k", 4))

    def _get_relevant_documents(self, query: str) -> List[LangchainDocument]:
        """Sync version of get_relevant_documents."""
        return asyncio.run(self._aget_relevant_documents(query))


class VectorStore:
    """Abstract base vector store interface."""

    def __init__(self, context_id: Optional[str] = None) -> None:
        self.embeddings = OpenAIEmbeddings()
        self.context_id = context_id
        self.vectorstore = self._initialize_vectorstore()

    def _initialize_vectorstore(self) -> PGVector:
        """Initialize the PGVector instance."""
        return PGVector(
            embeddings=self.embeddings,
            collection_name=f"{self.context_id}_document",
            connection=get_database_url()
        )

    def as_retriever(self, search_kwargs: Optional[dict] = None, context_id: Optional[str] = None) -> CustomRetriever:
        """Return a retriever interface with optional context isolation."""
        return CustomRetriever(
            vectorstore=self, search_kwargs=search_kwargs or {"k": 4}, context_id=context_id or self.context_id
        )

    async def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to the vector store with context isolation."""
        if metadatas is None:
            metadatas = [{}] * len(texts)

        documents = [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]

        try:
            await asyncio.get_event_loop().run_in_executor(None, self.vectorstore.add_documents, documents)
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            raise

    async def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> List[LangchainDocument]:
        """Perform similarity search with context filtering."""
        try:
            query_embedding = await self.embeddings.aembed_query(query)
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.vectorstore.similarity_search_by_vector, query_embedding, k
            )

            return [
                LangchainDocument(
                    page_content=doc.page_content,
                    metadata=doc.metadata if isinstance(doc.metadata, dict) else {},
                )
                for doc in results
            ]
        except Exception:
            return []

    async def set_context_filter(self, context_id: str) -> None:
        """Set the context filter for the vector store."""
        self.context_id = context_id

    async def clear_context_filter(self) -> None:
        """Clear the context filter for the vector store."""
        self.context_id = None

