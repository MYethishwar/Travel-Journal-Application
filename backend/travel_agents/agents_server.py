# =============================================
# agent_server.py — FastAPI server for agents
# Run: uvicorn agent_server:app --port 8001 --reload
# =============================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import run_travel_agents

app = FastAPI(title="Travel Multi-Agent API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str  # e.g. "Plan a 5 day trip to Tokyo"


@app.get("/")
def root():
    return {"status": "Travel Agent API running", "port": 8001}


@app.post("/agent/travel")
async def travel_agent(req: QueryRequest):
    """
    Main endpoint. Send any travel question.
    The multi-agent workflow runs and returns all outputs.
    """
    result = run_travel_agents(req.query)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}