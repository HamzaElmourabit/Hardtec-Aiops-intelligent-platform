"""HARDTEC agentic support service.

The agent coordinates existing HARDTEC capabilities instead of replacing them:
ticket classification, historical ticket retrieval, and action planning.
An external LLM is optional and disabled by default.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.predictor import predict_ticket

try:
    from src.rag.rag_engine import rag_assets_available, retrieve_similar_tickets
except Exception as exc:  # pragma: no cover - depends on local RAG assets
    retrieve_similar_tickets = None
    rag_assets_available = None
    RAG_IMPORT_ERROR = str(exc)
else:
    RAG_IMPORT_ERROR = None


app = FastAPI(
    title="HARDTEC Agent API",
    description="Agentic orchestration for ticket analysis and support actions.",
    version="1.0.0",
)


class AgentRequest(BaseModel):
    ticket_text: str = Field(min_length=5, max_length=10_000)
    top_k: int = Field(default=3, ge=1, le=10)
    allow_external_llm: bool = False


def _retrieve_context(ticket_text: str, top_k: int) -> list[dict[str, Any]]:
    if retrieve_similar_tickets is None:
        return []

    results = retrieve_similar_tickets(ticket_text, top_k=top_k)
    context = []
    for result in results:
        document = result.get("document", "") if isinstance(result, dict) else str(result)
        context.append(
            {
                "score": float(result.get("score", 0.0)) if isinstance(result, dict) else 0.0,
                "document": document,
            }
        )
    return context


def _build_action_plan(prediction: dict[str, Any], context: list[dict[str, Any]]) -> list[str]:
    priority = str(prediction.get("priority", "")).lower()
    queue = prediction.get("queue", "support")
    actions = [f"Affecter le ticket a la file {queue}."]

    if priority in {"high", "critical", "urgent"}:
        actions.insert(0, "Escalader immediatement le ticket a l'equipe d'astreinte.")
    else:
        actions.insert(0, "Accuser reception du ticket et planifier son traitement.")

    if context:
        actions.append("Comparer la solution proposee avec les tickets historiques similaires.")
    actions.append("Demander une validation humaine avant toute action irreversible.")
    return actions


async def _optional_llm_summary(
    ticket_text: str,
    prediction: dict[str, Any],
    context: list[dict[str, Any]],
) -> str | None:
    if not os.getenv("AGENT_LLM_URL") or not os.getenv("AGENT_LLM_API_KEY"):
        return None

    prompt = {
        "ticket": ticket_text,
        "prediction": prediction,
        "similar_tickets": context,
        "instruction": "Return a concise French support recommendation.",
    }
    headers = {"Authorization": f"Bearer {os.environ['AGENT_LLM_API_KEY']}"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(os.environ["AGENT_LLM_URL"], json=prompt, headers=headers)
        response.raise_for_status()
        payload = response.json()
    return str(payload.get("response") or payload.get("message") or payload.get("content", ""))


@app.get("/health")
def health() -> dict[str, Any]:
    rag_available = (
        retrieve_similar_tickets is not None
        and rag_assets_available is not None
        and rag_assets_available()
    )
    return {
        "status": "healthy",
        "service": "hardtec-agent",
        "rag_available": rag_available,
        "rag_error": None if rag_available else RAG_IMPORT_ERROR or "RAG model assets are missing.",
        "llm_configured": bool(os.getenv("AGENT_LLM_URL") and os.getenv("AGENT_LLM_API_KEY")),
    }


@app.post("/agent/analyze")
async def analyze(request: AgentRequest) -> dict[str, Any]:
    try:
        prediction = predict_ticket(request.ticket_text)
        context = _retrieve_context(request.ticket_text, request.top_k)
        actions = _build_action_plan(prediction, context)
        llm_summary = None
        if request.allow_external_llm:
            llm_summary = await _optional_llm_summary(request.ticket_text, prediction, context)

        return {
            "ticket_text": request.ticket_text,
            "prediction": prediction,
            "similar_tickets": context,
            "recommended_actions": actions,
            "summary": llm_summary or "Analyse locale realisee par les modeles HARDTEC.",
            "human_approval_required": True,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Agent analysis failed: {exc}") from exc