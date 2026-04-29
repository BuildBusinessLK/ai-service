from fastapi import FastAPI
from pydantic import BaseModel, Field
from rag.query import get_qa_chain
from typing import List, Optional

app = FastAPI()

qa_chain = get_qa_chain()

class ChatMessage(BaseModel):
    role: str
    content: str

class Query(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    chat_history: List[ChatMessage] = Field(default_factory=list)

@app.post("/ask")
def ask(query: Query):
    result = qa_chain.invoke(
        {
            "input": query.question,
            "chat_history": [message.dict() for message in query.chat_history],
        }
    )
    return {"answer": result["answer"]}