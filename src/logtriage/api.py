from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import tools

app = FastAPI(title="LogTriage Agent", version="0.1.0")


class TriageRequest(BaseModel):
    log_text: str = Field(min_length=1)
    backend: str = "offline"
    prompt_version: str = "v2"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/triage")
def triage(req: TriageRequest) -> dict:
    if req.prompt_version not in ("v1", "v2"):
        raise HTTPException(422, "prompt_version must be v1 or v2")
    try:
        return tools.triage_log(req.log_text, req.backend, req.prompt_version)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except KeyError as exc:
        raise HTTPException(400, f"missing configuration: {exc}")


@app.get("/defects/search")
def defects_search(q: str, k: int = 3) -> list[dict]:
    return tools.search_defects(q, k)
