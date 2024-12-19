from fastapi import FastAPI
from src.routes.rag import router as rag_router

app = FastAPI()

app.include_router(rag_router, prefix="/rag")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)





