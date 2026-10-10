import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.security import (
    cors_origins,
    enforce_api_key,
    validate_api_configuration,
)


validate_api_configuration()


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="HARDTEC AI Operations & Intelligent Support API",
    description=(
        "AI-powered IT incident prediction, intelligent ticketing, "
        "AIOps risk prediction and historical support recommendation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Apply authentication, request limits, and browser security headers."""
    content_length = request.headers.get("content-length")
    try:
        request_size = int(content_length) if content_length else 0
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid Content-Length header."},
        )

    if request_size > 1_048_576:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body is too large."},
        )

    try:
        enforce_api_key(request)
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


# ============================================================
# TICKET ML
# ============================================================

try:
    from src.pipeline.predictor import predict_ticket

    TICKET_ML_AVAILABLE = True
    TICKET_ML_ERROR = None

except Exception as e:
    TICKET_ML_AVAILABLE = False
    TICKET_ML_ERROR = str(e)

    def predict_ticket(ticket_text: str):
        raise RuntimeError(f"Ticket ML unavailable: {e}")


# ============================================================
# RAG
# ============================================================

try:
    from src.rag.rag_engine import rag_assets_available, retrieve_similar_tickets

    RAG_AVAILABLE = rag_assets_available()
    RAG_ERROR = None if RAG_AVAILABLE else "RAG model assets are missing."

except Exception as e:
    RAG_AVAILABLE = False
    RAG_ERROR = str(e)

    def retrieve_similar_tickets(query: str, top_k: int = 3):
        raise RuntimeError(f"RAG unavailable: {e}")


# ============================================================
# SQLITE DATABASE
# ============================================================

try:
    from src.data.ticket_store import TicketStore

    ticket_store = TicketStore()

    DATABASE_AVAILABLE = True
    DATABASE_ERROR = None

except Exception as e:
    ticket_store = None
    DATABASE_AVAILABLE = False
    DATABASE_ERROR = str(e)


# ============================================================
# AIOPS
# ============================================================

AIOPS_AVAILABLE = True
AIOPS_ERROR = None
_aiops_predictor = None


def load_aiops():
    """
    Lazy-load the MicroSS AIOps predictor.

    Actual implementation:
        src.aiops.micross.risk_predictor.MicroSSRiskPredictor

    Prediction:
        predictor.predict(timestamp)
    """

    global _aiops_predictor
    global AIOPS_AVAILABLE
    global AIOPS_ERROR

    if _aiops_predictor is not None:
        return _aiops_predictor

    try:
        from src.aiops.micross.risk_predictor import MicroSSRiskPredictor

        _aiops_predictor = MicroSSRiskPredictor()

        AIOPS_AVAILABLE = True
        AIOPS_ERROR = None

        return _aiops_predictor

    except Exception as e:
        AIOPS_AVAILABLE = False
        AIOPS_ERROR = str(e)

        raise RuntimeError(
            f"AIOps predictor unavailable: {e}"
        )


# ============================================================
# RAG TEXT PARSER
# ============================================================

def parse_rag_text(document: str) -> Dict[str, Optional[str]]:
    """
    Parse metadata embedded in historical ticket documents.

    Expected format:

        Type: Incident
        Queue: IT Support
        Priority: high
        Language: en
        Historical solution: ...
    """

    result = {
        "type": None,
        "queue": None,
        "priority": None,
        "language": None,
        "historical_solution": None,
    }

    if not isinstance(document, str):
        return result

    patterns = {
        "type": r"Type:\s*(.+?)(?=\n|$)",
        "queue": r"Queue:\s*(.+?)(?=\n|$)",
        "priority": r"Priority:\s*(.+?)(?=\n|$)",
        "language": r"Language:\s*(.+?)(?=\n|$)",
        "historical_solution": (
            r"Historical solution:\s*(.+?)(?=\n|$)"
        ),
    }

    for key, pattern in patterns.items():

        match = re.search(
            pattern,
            document,
            flags=re.IGNORECASE,
        )

        if match:
            value = match.group(1).strip()

            if value:
                result[key] = value

    return result


# ============================================================
# PYDANTIC MODELS
# ============================================================

class TicketRequest(BaseModel):
    ticket_text: str = Field(
        ...,
        min_length=1,
        description="IT support ticket description",
    )


class AIOpsRiskRequest(BaseModel):
    timestamp: str = Field(
        ...,
        min_length=1,
        description="MicroSS timestamp used for AIOps prediction",
    )


class RAGSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Search query for historical support tickets",
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of historical tickets to retrieve",
    )


class SupportAnalyzeRequest(BaseModel):
    ticket_text: str = Field(
        ...,
        min_length=1,
        description="Current IT support ticket",
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of historical tickets to retrieve",
    )

    micross_timestamp: Optional[str] = Field(
        default=None,
        description="Optional MicroSS timestamp for AIOps risk prediction",
    )


# ============================================================
# RECOMMENDATION
# ============================================================

def generate_recommendation(
    ticket_type: Optional[str],
    queue: Optional[str],
    priority: Optional[str],
) -> str:

    if queue and priority:
        return (
            f"Assign the ticket to the {queue} team with "
            f"{priority} priority and monitor the incident "
            "through its resolution lifecycle."
        )

    if queue:
        return (
            f"Assign the ticket to the {queue} team and monitor "
            "the incident through its resolution lifecycle."
        )

    if priority:
        return (
            f"Handle the ticket with {priority} priority and "
            "monitor it through its resolution lifecycle."
        )

    return (
        "Review the ticket and assign it to the appropriate "
        "support team for resolution."
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "application": (
            "HARDTEC AI Operations & Intelligent Support API"
        ),
        "version": "1.0.0",
        "status": "running",
        "services": {
            "ticket_ml": TICKET_ML_AVAILABLE,
            "aiops": AIOPS_AVAILABLE,
            "rag": RAG_AVAILABLE,
            "database": DATABASE_AVAILABLE,
        },
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    services = {
        "ticket_ml": (
            "available"
            if TICKET_ML_AVAILABLE
            else "unavailable"
        ),
        "aiops": (
            "available"
            if AIOPS_AVAILABLE
            else "unavailable"
        ),
        "rag": (
            "available"
            if RAG_AVAILABLE
            else "unavailable"
        ),
        "database": (
            "SQLite"
            if DATABASE_AVAILABLE
            else "unavailable"
        ),
        "llm": "optional / not configured",
    }

    errors = {
        "ticket_ml": TICKET_ML_ERROR,
        "aiops": AIOPS_ERROR,
        "rag": RAG_ERROR,
        "database": DATABASE_ERROR,
    }

    core_services_available = (
        TICKET_ML_AVAILABLE
        and AIOPS_AVAILABLE
        and RAG_AVAILABLE
        and DATABASE_AVAILABLE
    )

    return {
        "status": (
            "healthy"
            if core_services_available
            else "degraded"
        ),
        "application": (
            "HARDTEC AI Operations & Intelligent Support API"
        ),
        "services": services,
        "errors": errors,
    }


# ============================================================
# TICKET PREDICTION
# ============================================================

@app.post("/predict")
def predict_ticket_endpoint(request: TicketRequest):

    if not TICKET_ML_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Ticket ML service is unavailable.",
        )

    try:

        prediction = predict_ticket(
            request.ticket_text
        )

        ticket_type = prediction.get(
            "ticket_type"
        )

        priority = prediction.get(
            "priority"
        )

        queue = prediction.get(
            "queue"
        )

        confidence = prediction.get(
            "confidence",
            0.85,
        )

        recommendation = generate_recommendation(
            ticket_type=ticket_type,
            queue=queue,
            priority=priority,
        )

        prediction_id = None
        save_status = "not_saved"

        # ----------------------------------------------------
        # SQLITE PERSISTENCE
        # ----------------------------------------------------

        if DATABASE_AVAILABLE:

            prediction_id = ticket_store.save_prediction(
                ticket_text=request.ticket_text,
                ticket_type=ticket_type,
                priority=priority,
                queue=queue,
                confidence=confidence,
                recommendation=recommendation,
                source="api",
            )

            save_status = "saved"

        return {
            "ticket_text": request.ticket_text,
            "type": ticket_type,
            "priority": priority,
            "queue": queue,
            "confidence": confidence,
            "recommendation": recommendation,
            "prediction_id": prediction_id,
            "save_status": save_status,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


# ============================================================
# AIOPS RISK PREDICTION
# ============================================================

@app.post("/aiops/predict-risk")
def aiops_predict_risk(
    request: AIOpsRiskRequest,
):

    if not AIOPS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AIOps service is unavailable.",
        )

    try:

        aiops_predictor = load_aiops()

        # MicroSSRiskPredictor exposes .predict(timestamp)
        result = aiops_predictor.predict(
            request.timestamp
        )

        return {
            "timestamp": request.timestamp,
            "aiops": result,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "AIOps risk prediction failed: "
                f"{str(e)}"
            ),
        )


# ============================================================
# RAG SEARCH
# ============================================================

@app.post("/rag/search")
def rag_search(
    request: RAGSearchRequest,
):

    if not RAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="RAG service is unavailable.",
        )

    try:

        results = retrieve_similar_tickets(
            request.query,
            top_k=request.top_k,
        )

        formatted_results: List[
            Dict[str, Any]
        ] = []

        for item in results:

            if isinstance(item, dict):

                document = item.get(
                    "document",
                    item.get("text", ""),
                )

                score = item.get(
                    "score",
                    item.get("similarity"),
                )

                parsed = parse_rag_text(
                    document
                    if isinstance(document, str)
                    else ""
                )

            else:

                document = str(item)
                score = None

                parsed = parse_rag_text(
                    document
                )

            formatted_results.append(
                {
                    "score": score,
                    "ticket": document,
                    "type": parsed["type"],
                    "queue": parsed["queue"],
                    "priority": parsed["priority"],
                    "language": parsed["language"],
                    "historical_solution": (
                        parsed["historical_solution"]
                    ),
                }
            )

        return {
            "query": request.query,
            "top_k": request.top_k,
            "results": formatted_results,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"RAG search failed: {str(e)}",
        )


# ============================================================
# INTELLIGENT SUPPORT ANALYSIS
# ============================================================

@app.post("/support/analyze")
def support_analyze(
    request: SupportAnalyzeRequest,
):

    if not TICKET_ML_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Ticket ML service is unavailable.",
        )

    if not RAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="RAG service is unavailable.",
        )

    try:

        # ----------------------------------------------------
        # 1. TICKET ML
        # ----------------------------------------------------

        prediction = predict_ticket(
            request.ticket_text
        )

        ticket_type = prediction.get(
            "ticket_type"
        )

        priority = prediction.get(
            "priority"
        )

        queue = prediction.get(
            "queue"
        )

        confidence = prediction.get(
            "confidence",
            0.85,
        )

        # ----------------------------------------------------
        # 2. RAG
        # ----------------------------------------------------

        rag_results = retrieve_similar_tickets(
            request.ticket_text,
            top_k=request.top_k,
        )

        historical_cases: List[
            Dict[str, Any]
        ] = []

        for item in rag_results:

            if isinstance(item, dict):

                document = item.get(
                    "document",
                    item.get("text", ""),
                )

                score = item.get(
                    "score",
                    item.get("similarity"),
                )

            else:

                document = str(item)
                score = None

            parsed = parse_rag_text(
                document
                if isinstance(document, str)
                else ""
            )

            historical_cases.append(
                {
                    "score": score,
                    "ticket": document,
                    "type": parsed["type"],
                    "queue": parsed["queue"],
                    "priority": parsed["priority"],
                    "language": parsed["language"],
                    "historical_solution": (
                        parsed["historical_solution"]
                    ),
                }
            )

        # ----------------------------------------------------
        # 3. AIOPS
        # ----------------------------------------------------

        aiops_result = None

        if request.micross_timestamp:

            if not AIOPS_AVAILABLE:
                raise HTTPException(
                    status_code=503,
                    detail="AIOps service is unavailable.",
                )

            aiops_predictor = load_aiops()

            aiops_result = aiops_predictor.predict(
                request.micross_timestamp
            )

        # ----------------------------------------------------
        # 4. RECOMMENDATION
        # ----------------------------------------------------

        recommendation = generate_recommendation(
            ticket_type=ticket_type,
            queue=queue,
            priority=priority,
        )

        # ----------------------------------------------------
        # 5. FINAL RESPONSE
        # ----------------------------------------------------

        return {
            "ticket_text": request.ticket_text,

            "prediction": {
                "type": ticket_type,
                "queue": queue,
                "priority": priority,
                "confidence": confidence,
            },

            "aiops": aiops_result,

            "recommendation": recommendation,

            "historical_cases": historical_cases,
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Support analysis failed: "
                f"{str(e)}"
            ),
        )


# ============================================================
# HISTORY
# ============================================================

@app.get("/history")
def history(
    limit: int = 50,
):

    if not DATABASE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Database service is unavailable.",
        )

    if limit < 1:
        raise HTTPException(
            status_code=400,
            detail="limit must be greater than 0.",
        )

    try:

        records = ticket_store.get_recent_predictions(
            limit=limit
        )

        return {
            "count": len(records),
            "records": records,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to retrieve history: {str(e)}"
            ),
        )


# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.get("/history/{prediction_id}")
def history_by_id(
    prediction_id: int,
):

    if not DATABASE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Database service is unavailable.",
        )

    try:

        record = ticket_store.get_prediction_by_id(
            prediction_id
        )

        if record is None:

            raise HTTPException(
                status_code=404,
                detail="Prediction not found.",
            )

        return record

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to retrieve prediction: "
                f"{str(e)}"
            ),
        )


# ============================================================
# STATISTICS
# ============================================================

@app.get("/stats")
def statistics():

    if not DATABASE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Database service is unavailable.",
        )

    try:

        return ticket_store.get_stats()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to retrieve statistics: {str(e)}"
            ),
        )


# ============================================================
# RUNNING
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )