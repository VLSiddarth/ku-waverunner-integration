"""
ku_client.py
Knowledge Universe API client for KU + Waverunner integration.
Author: V.L. Siddarth / Rick (pair session)
"""

import os
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

KU_BASE_URL = os.getenv("KU_BASE_URL", "https://vlsiddarth-knowledge-universe.hf.space")
KU_API_KEY  = os.getenv("KU_API_KEY", "")


class KUClient:
    """Thin, authenticated wrapper around the Knowledge Universe /v1/discover endpoint."""

    def __init__(self, api_key: str = KU_API_KEY, base_url: str = KU_BASE_URL):
        if not api_key:
            raise ValueError("KU_API_KEY not set. Check your .env file.")
        self.api_key  = api_key
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()
        self.session.headers.update({
            "X-API-Key":    self.api_key,
            "Content-Type": "application/json",
        })

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    def health(self) -> dict:
        resp = self.session.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Core discover call
    # ------------------------------------------------------------------
    def discover(
        self,
        topic:       str,
        difficulty:  int  = 3,
        max_results: int  = 20,
        domain:      Optional[str] = None,
    ) -> dict:
        """
        POST /v1/discover — returns full KU response including
        decay_scores, knowledge_velocity, conflict_detection, etc.
        """
        payload = {
            "topic":       topic,
            "difficulty":  difficulty,
            "max_results": max_results,
        }
        if domain:
            payload["domain"] = domain

        resp = self.session.post(
            f"{self.base_url}/v1/discover",
            json=payload,
            timeout=60,   # cold queries can take ~30 s
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Convenience: extract just the decay scores dict
    # ------------------------------------------------------------------
    def get_decay_scores(self, ku_response: dict) -> dict:
        """Returns {source_id: decay_report_dict} from a discover response."""
        return ku_response.get("decay_scores", {})

    # ------------------------------------------------------------------
    # Convenience: extract sources list
    # ------------------------------------------------------------------
    def get_sources(self, ku_response: dict) -> list:
        return ku_response.get("sources", [])


# ------------------------------------------------------------------
# Quick smoke test — run directly: python ku_client.py
# ------------------------------------------------------------------
if __name__ == "__main__":
    from rich.console import Console
    from rich.table   import Table

    console = Console()
    client  = KUClient()

    console.print("\n[bold]KU health check...[/bold]")
    health = client.health()
    console.print(f"  Status:  {health.get('status')}")
    console.print(f"  Version: {health.get('version')}")
    console.print(f"  Redis:   {health.get('redis')}\n")

    console.print("[bold]Running test discover query...[/bold]")
    result = client.discover(
        topic="LLM agent deployment enterprise environments",
        difficulty=3,
        max_results=5,
    )

    console.print(f"  total_found:       {result.get('total_found')}")
    console.print(f"  cache_hit:         {result.get('cache_hit')}")
    console.print(f"  processing_time:   {result.get('processing_time_ms', 0):.0f} ms")
    console.print(f"  credits_remaining: {result.get('credits_remaining')}")
    console.print(f"  knowledge_velocity:{result.get('knowledge_velocity', {}).get('velocity_label')}\n")

    table = Table(title="Decay scores", show_lines=True)
    table.add_column("Source ID",      style="dim",    max_width=40)
    table.add_column("decay_score",    justify="right")
    table.add_column("label",          justify="center")
    table.add_column("age_days",       justify="right")
    table.add_column("days_until_stale", justify="right")

    for src_id, dr in client.get_decay_scores(result).items():
        table.add_row(
            src_id[:40],
            f"{dr.get('decay_score', 0):.3f}",
            dr.get("label", "?"),
            str(dr.get("age_days", "?")),
            str(dr.get("days_until_stale", "?")),
        )
    console.print(table)