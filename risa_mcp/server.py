from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLINICAL_GUIDANCE_PATH = PROJECT_ROOT / "knowledge" / "clinical_guidance.json"
LEARNED_GUIDANCE_ADMIN_TOKEN = os.getenv("LEARNED_GUIDANCE_ADMIN_TOKEN", "").strip()


def _api_base() -> str:
    base = os.getenv("RISA_API_BASE") or os.getenv("RISA_API_BASE") or "http://127.0.0.1:8000"
    return base.rstrip("/")


def _load_guidance() -> Dict[str, Any]:
    try:
        with CLINICAL_GUIDANCE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {"sources": [], "topics": []}


async def _request_json(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{_api_base()}{path}"
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(method, url, json=payload)
        response.raise_for_status()
        return response.json()


async def _request_json_with_headers(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    url = f"{_api_base()}{path}"
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(method, url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


def _topic_summary(topic: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": topic.get("id"),
        "title": topic.get("title"),
        "keywords": topic.get("keywords", []),
        "core_guidance": topic.get("core_guidance", []),
        "red_flags": topic.get("red_flags", []),
        "support_steps": topic.get("support_steps", []),
    }


def _find_topic(topic_id: str) -> Optional[Dict[str, Any]]:
    guidance = _load_guidance()
    for topic in guidance.get("topics", []):
        if topic.get("id") == topic_id:
            return topic
    return None


def _admin_headers() -> Dict[str, str]:
    if LEARNED_GUIDANCE_ADMIN_TOKEN:
        return {"X-Admin-Token": LEARNED_GUIDANCE_ADMIN_TOKEN}
    return {}


mcp = FastMCP("RISA MCP")


@mcp.tool()
async def health() -> Dict[str, Any]:
    """Check whether the RISA backend is reachable."""
    return await _request_json("GET", "/health")


@mcp.tool()
async def stats() -> Dict[str, Any]:
    """Get backend and RAG status for the RISA project."""
    return await _request_json("GET", "/stats")


@mcp.tool()
async def ask_risa(message: str, language: Optional[str] = None, location: Optional[str] = None, device_id: Optional[str] = None) -> Dict[str, Any]:
    """Send a message to RISA and get the direct response payload."""
    payload: Dict[str, Any] = {
        "message": message,
        "location": location,
        "device_id": device_id,
    }
    if language:
        payload["source_lang"] = language
    return await _request_json("POST", "/chat", payload)


@mcp.tool()
async def clear_session(device_id: Optional[str] = None) -> Dict[str, Any]:
    """Clear the in-memory conversation state for a device."""
    return await _request_json("POST", "/clear", {"message": "", "device_id": device_id})


@mcp.tool()
async def clinical_topics() -> Dict[str, Any]:
    """List the clinical guidance topics used by RISA."""
    guidance = _load_guidance()
    return {
        "sources": guidance.get("sources", []),
        "topics": [_topic_summary(topic) for topic in guidance.get("topics", [])],
    }


@mcp.tool()
async def clinical_topic(topic_id: str) -> Dict[str, Any]:
    """Get one clinical guidance topic by id."""
    topic = _find_topic(topic_id)
    if not topic:
        return {"error": f"Unknown topic_id: {topic_id}", "available_topics": [t.get("id") for t in _load_guidance().get("topics", [])]}
    return _topic_summary(topic)


@mcp.tool()
async def learned_topics(limit: int = 10) -> Dict[str, Any]:
    """Inspect the latest learned guidance topics from conversation self-learning."""
    return await _request_json_with_headers("GET", f"/learned-topics?limit={limit}", headers=_admin_headers())


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
