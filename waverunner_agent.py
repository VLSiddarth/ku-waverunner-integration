"""
waverunner_agent.py

Waverunner agent: Gemini LLM + Knowledge Universe temporal decay gate.
Image 1 integration — agent calls KU, decay gate filters stale documents,
Gemini synthesises over only fresh, validated context.

Usage:
  python waverunner_agent.py "FDA clinical trial reporting requirements 2026"
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv
from rich.console  import Console
from rich.panel    import Panel
from rich.markdown import Markdown

from ku_client  import KUClient
from decay_gate import DecayGate, GateResult

load_dotenv()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
DECAY_THRESHOLD = float(os.getenv("DECAY_THRESHOLD", "0.40"))
GEMINI_MODEL    = "gemini-2.0-flash"

console = Console()


# ------------------------------------------------------------------
# Response dataclass
# ------------------------------------------------------------------
@dataclass
class AgentResponse:
    query:              str
    answer:             str
    gate_result:        object
    model_used:         str
    latency_ms:         float
    context_tokens_est: int
    tokens_saved:       int
    error:              Optional[str] = None


# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a Waverunner enterprise AI agent specialising in clinical NLP and
regulatory compliance. You operate with a temporal decay governance layer
(Knowledge Universe API v2.2) that ensures every document in your context
has passed a freshness gate (decay_score <= {threshold}).

Rules:
1. Synthesise ONLY from the KU context below — do not use training knowledge
   for specific facts, protocols, or guidelines. It may be outdated.
2. If a document is labeled 'aging', note it will expire soon.
3. If no documents were admitted, say so — do not hallucinate.
4. Always cite document title and platform for every factual claim.
5. Flag any conflicts detected by the KU conflict_detection layer.

Conflict status:    {conflict_status}
Knowledge velocity: {velocity_label} — {velocity_note}

{context}
"""


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------
class WaverunnerAgent:
    """KU-gated Gemini agent. Each .run() call is fully stateless."""

    def __init__(
        self,
        gemini_api_key:  str   = GEMINI_API_KEY,
        decay_threshold: float = DECAY_THRESHOLD,
        ku_difficulty:   int   = 3,
        ku_max_results:  int   = 20,
        tokens_per_doc:  int   = 800,
        model_name:      str   = GEMINI_MODEL,
    ):
        if not gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. "
                "Get one at https://aistudio.google.com then add to .env"
            )
        self.client          = genai.Client(api_key=gemini_api_key)
        self.ku_client       = KUClient()
        self.decay_gate      = DecayGate(threshold=decay_threshold, tokens_per_doc=tokens_per_doc)
        self.ku_difficulty   = ku_difficulty
        self.ku_max_results  = ku_max_results
        self.decay_threshold = decay_threshold
        self.model_name      = model_name

    def run(
        self,
        query:   str,
        domain:  Optional[str] = None,
        verbose: bool = True,
    ) -> AgentResponse:

        t_start = time.time()

        if verbose:
            console.print(f"\n[bold cyan]Waverunner agent[/bold cyan] — {query!r}")

        # 1. KU discover
        if verbose:
            console.print("  [dim]-> Calling KU /v1/discover...[/dim]")
        try:
            ku_resp = self.ku_client.discover(
                topic=query,
                difficulty=self.ku_difficulty,
                max_results=self.ku_max_results,
                domain=domain,
            )
        except Exception as exc:
            return AgentResponse(
                query=query, answer="", gate_result=None,
                model_used=self.model_name, latency_ms=0,
                context_tokens_est=0, tokens_saved=0,
                error=f"KU API error: {exc}",
            )

        if verbose:
            console.print(
                f"  [dim]  total_found={ku_resp.get('total_found')} "
                f"cache_hit={ku_resp.get('cache_hit')} "
                f"processing={ku_resp.get('processing_time_ms', 0):.0f}ms[/dim]"
            )

        # 2. Decay gate
        if verbose:
            console.print(
                f"  [dim]-> Decay gate (threshold={self.decay_threshold})...[/dim]"
            )

        gate_result = self.decay_gate.apply(ku_resp, topic=query)

        if verbose:
            console.print(
                f"  [dim]  admitted={gate_result.admitted_count} "
                f"blocked={gate_result.blocked_count} "
                f"tokens_saved=~{gate_result.tokens_saved:,}[/dim]"
            )
            for doc in gate_result.blocked_docs:
                console.print(
                    f"  [red]  GATED:[/red] {doc.title[:60]} "
                    f"(decay={doc.decay_score:.3f})"
                )
            for doc in gate_result.admitted_docs:
                console.print(
                    f"  [green]  ADMITTED:[/green] {doc.title[:60]} "
                    f"(decay={doc.decay_score:.3f})"
                )

        # 3. Build system prompt
        vel  = gate_result.knowledge_velocity
        conf = gate_result.conflict_detection

        system_prompt = SYSTEM_PROMPT.format(
            threshold       = self.decay_threshold,
            conflict_status = (
                f"{conf.get('conflicts_found', 0)} conflict(s) detected"
                if conf.get("conflicts_found") else "No conflicts detected"
            ),
            velocity_label  = vel.get("velocity_label", "unknown").upper(),
            velocity_note   = vel.get("warning") or vel.get("interpretation", ""),
            context         = gate_result.context_text,
        )

        # 4. Gemini
        if verbose:
            console.print(f"  [dim]-> Sending to Gemini ({self.model_name})...[/dim]")

        try:
            response = self.client.models.generate_content(
                model    = self.model_name,
                contents = f"User query: {query}",
                config   = types.GenerateContentConfig(
                    system_instruction = system_prompt,
                    temperature        = 0.2,
                    max_output_tokens  = 1024,
                ),
            )
            answer = response.text
        except Exception as exc:
            return AgentResponse(
                query=query, answer="", gate_result=gate_result,
                model_used=self.model_name,
                latency_ms=(time.time() - t_start) * 1000,
                context_tokens_est=len(gate_result.context_text.split()),
                tokens_saved=gate_result.tokens_saved,
                error=f"Gemini error: {exc}",
            )

        latency_ms = (time.time() - t_start) * 1000

        if verbose:
            console.print(f"  [dim]  Done — {latency_ms:.0f}ms total[/dim]")

        return AgentResponse(
            query              = query,
            answer             = answer,
            gate_result        = gate_result,
            model_used         = self.model_name,
            latency_ms         = latency_ms,
            context_tokens_est = len(gate_result.context_text.split()),
            tokens_saved       = gate_result.tokens_saved,
        )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "best practices for deploying LLM agents in enterprise environments"
    )

    agent = WaverunnerAgent()
    resp  = agent.run(query, verbose=True)

    if resp.error:
        console.print(f"\n[red]Error:[/red] {resp.error}")
        sys.exit(1)

    console.print("\n")
    console.print(Panel(
        Markdown(resp.answer),
        title    = f"[bold]Waverunner agent[/bold] -- {resp.model_used}",
        subtitle = (
            f"latency={resp.latency_ms:.0f}ms | "
            f"admitted={resp.gate_result.admitted_count} docs | "
            f"tokens_saved=~{resp.tokens_saved:,}"
        ),
    ))

    gate = resp.gate_result
    console.print(
        f"\n[dim]Gate: {gate.admitted_count} admitted | "
        f"{gate.blocked_count} blocked | "
        f"~{gate.tokens_saved:,} tokens eliminated before billing[/dim]"
    )


if __name__ == "__main__":
    main()