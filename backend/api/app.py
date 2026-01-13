from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.ragEngine import ragEngine, search, chat
import os
import dotenv

dotenv.load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL")],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, "data", "hr.pdf")
GEMINI_API_KEY = os.getenv("API_KEY")

ragState = ragEngine(PDF_PATH, GEMINI_API_KEY)


class Query(BaseModel):
    query: str


@app.post("/search")
def search_docs(q: Query):
    results = search(
        q.query,
        ragState["embedder"],
        ragState["index"],
        ragState["chunks"],
        ragState["pages"],
    )

    return {
        "results": [
            {"text": chunk, "page": page}
            for chunk, page in results
        ]
    }

@app.post("/chat")
def chat_with_docs(q: Query):
    answer, refs = chat(q.query, ragState)

    return {
        "answer": answer,
        "sources": [{"page": page} for _, page in refs]
    }


@app.get("/")
def healthCheck():
    return{"Health": "OK"}