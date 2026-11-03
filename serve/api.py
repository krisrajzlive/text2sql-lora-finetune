"""REST service: POST /generate {question, schema} -> {sql}."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER = os.environ.get("ADAPTER_ID", "")


class GenerateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    schema_sql: str = Field(min_length=1, max_length=5000, alias="schema")


class GenerateResponse(BaseModel):
    sql: str


@asynccontextmanager
async def lifespan(app):
    if getattr(app.state, "model", None) is None:  # tests inject a stub
        if not ADAPTER:
            raise RuntimeError("Set ADAPTER_ID to a Hub repo id or local adapter path.")
        from src.inference import Text2SQL

        app.state.model = Text2SQL(BASE_MODEL, ADAPTER)
    yield


app = FastAPI(title="text2sql-lora", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "base_model": BASE_MODEL, "adapter": ADAPTER}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, request: Request):
    try:
        return GenerateResponse(sql=request.app.state.model.generate(req.question, req.schema_sql))
    except Exception as exc:  # surface model errors as a 500 with a message
        raise HTTPException(status_code=500, detail=str(exc))
