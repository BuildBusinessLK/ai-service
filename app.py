from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from rag.query import get_qa_chain
from typing import List, Optional

load_dotenv()

app = FastAPI()

qa_chain = get_qa_chain()

class ChatMessage(BaseModel):
    role: str
    content: str

class Query(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    chat_history: List[ChatMessage] = Field(default_factory=list)
    user_context: Optional[str] = None


def _message_to_dict(message: ChatMessage) -> dict:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return message.dict()


@app.post("/ask")
def ask(query: Query):
    result = qa_chain.invoke(
        {
            "input": query.question,
            "chat_history": [_message_to_dict(m) for m in query.chat_history],
            "user_context": (query.user_context or "").strip(),
        }
    )
    return {"answer": result["answer"]}