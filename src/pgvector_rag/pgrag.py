from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from functools import lru_cache

from .vector_store import VectorStore
from config import Settings
from ..schemas.rag import RAGConfig

SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions based on the context provided.
If you cannot find a relevant answer in the context, respond with "I don't know" - do not try to make up an answer.
Use the following format:
Question: [user's question]
Context: [retrieved documents]
Answer: [your response based strictly on the context and create a good sentance, or "I don't know" if no relevant information is found]"""


@lru_cache()
def get_settings() -> Settings:
    return Settings()


class PgRAGSystem:
    def __init__(self, context_id: str, config: RAGConfig) -> None:
        self.config = config
        self.vector_store = VectorStore(context_id)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        self.llm = ChatOpenAI(model_name=self.config.model_name, **self.config.model_config)
        self.prompt = PromptTemplate(
            template=f"{SYSTEM_PROMPT}\nQuestion: {{question}}\nContext: {{context}}\nAnswer: ",
            input_variables=["question", "context"],
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": self.config.top_k}),
            memory=self.memory,
            return_source_documents=True,
            output_key="answer",
            combine_docs_chain_kwargs={"prompt": self.prompt},
        )
        

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.

        Args:
            question: The question to be answered

        Returns:
            Dict containing answer and source documents

        Raises:
            ValueError: If there's a mismatch in embedding types
        """

        response = await self.chain.ainvoke({"question": question})
        return {"answer": response.get("answer"), "source_documents": response.get("source_documents")}
