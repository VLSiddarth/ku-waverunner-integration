"""
decay_gate.py
Temporal decay gate — sits between KU /v1/discover and the agent context window.
Hard-gates documents whose decay_score exceeds the configured threshold.
This is the Image 1 middleware layer described to the Waverunner team.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

DEFAULT_THRESHOLD = float(os.getenv("DECAY_THRESHOLD", "0.40"))


@dataclass
class GatedDocument:
    """A source document annotated with its gate decision."""
    source_id:        str
    title:            str
    url:              str
    platform:         str
    publication_date: Optional[str]
    summary:          str
    quality_score:    float
    peer_reviewed:    bool

    # Decay fields
    decay_score:      float
    freshness:        float
    age_days:         int
    label:            str            # fresh | aging | stale | decayed
    penalty_multiplier: float
    decay_velocity:   float
    days_until_stale: int

    # Gate decision
    admitted:         bool = True
    gate_reason:      str  = ""

    # Token estimate
    estimated_tokens: int = 0


@dataclass
class GateResult:
    """Full output of the decay gate for one query."""
    topic:            str
    threshold:        float
    total_retrieved:  int
    admitted_count:   int
    blocked_count:    int
    tokens_saved:     int           # estimated tokens never sent to the LLM
    avg_decay_score:  float
    knowledge_velocity: dict        = field(default_factory=dict)
    conflict_detection: dict        = field(default_factory=dict)
    admitted_docs:    list          = field(default_factory=list)   # list[GatedDocument]
    blocked_docs:     list          = field(default_factory=list)   # list[GatedDocument]
    context_text:     str           = ""    # ready-to-inject context string


class DecayGate:
    """
    Intercepts KU discover responses and applies a hard freshness gate.

    Usage:
        gate   = DecayGate(threshold=0.40, tokens_per_doc=800)
        result = gate.apply(ku_response, topic="FDA clinical protocols 2026")
        # result.context_text  → paste directly into LLM system prompt
        # result.tokens_saved  → tokens that never hit the API
    """

    def __init__(
        self,
        threshold:     float = DEFAULT_THRESHOLD,
        tokens_per_doc: int  = 800,
    ):
        self.threshold     = threshold
        self.tokens_per_doc = tokens_per_doc

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def apply(self, ku_response: dict, topic: str = "") -> GateResult:
        sources      = ku_response.get("sources", [])
        decay_scores = ku_response.get("decay_scores", {})

        admitted, blocked = [], []

        for src in sources:
            sid = src.get("id", "")
            dr  = decay_scores.get(sid) or src.get("decay_report") or {}

            doc = self._build_doc(src, dr)
            doc.estimated_tokens = self._estimate_tokens(src)

            if doc.decay_score > self.threshold:
                doc.admitted   = False
                doc.gate_reason = (
                    f"decay_score {doc.decay_score:.3f} exceeds threshold {self.threshold:.2f} "
                    f"[{doc.label}, age={doc.age_days}d]"
                )
                blocked.append(doc)
            else:
                doc.admitted = True
                admitted.append(doc)

        tokens_saved = sum(d.estimated_tokens for d in blocked)
        all_scores   = [d.decay_score for d in admitted + blocked]
        avg_decay    = sum(all_scores) / len(all_scores) if all_scores else 0.0

        context_text = self._build_context(admitted, topic)

        return GateResult(
            topic              = topic,
            threshold          = self.threshold,
            total_retrieved    = len(sources),
            admitted_count     = len(admitted),
            blocked_count      = len(blocked),
            tokens_saved       = tokens_saved,
            avg_decay_score    = avg_decay,
            knowledge_velocity = ku_response.get("knowledge_velocity", {}),
            conflict_detection = ku_response.get("conflict_detection", {}),
            admitted_docs      = admitted,
            blocked_docs       = blocked,
            context_text       = context_text,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_doc(self, src: dict, dr: dict) -> GatedDocument:
        return GatedDocument(
            source_id         = src.get("id", ""),
            title             = src.get("title", "Untitled"),
            url               = src.get("url", ""),
            platform          = src.get("source_platform", ""),
            publication_date  = (src.get("publication_date") or "")[:10],
            summary           = src.get("summary", "")[:400],
            quality_score     = src.get("quality_score", 0.0),
            peer_reviewed     = src.get("peer_reviewed", False),
            decay_score       = dr.get("decay_score", 1.0),
            freshness         = dr.get("freshness", 0.0),
            age_days          = dr.get("age_days", 9999),
            label             = dr.get("label", "unknown"),
            penalty_multiplier = dr.get("penalty_multiplier", 0.0),
            decay_velocity    = dr.get("decay_velocity", 0.0),
            days_until_stale  = dr.get("days_until_stale", 0),
        )

    def _estimate_tokens(self, src: dict) -> int:
        """Rough token estimate for a document entering the context window."""
        title_tok   = len((src.get("title", "") or "").split()) * 1.3
        summary_tok = len((src.get("summary", "") or "").split()) * 1.3
        meta_tok    = 40   # date, url, platform, etc.
        return int(title_tok + summary_tok + meta_tok) or self.tokens_per_doc

    def _build_context(self, admitted: list, topic: str) -> str:
        """
        Builds the context string injected into the LLM system prompt.
        Only admitted (fresh) documents appear here.
        """
        if not admitted:
            return (
                f"[KU DECAY GATE] No documents passed the freshness gate "
                f"(threshold={self.threshold:.2f}) for topic: '{topic}'. "
                "Do not synthesise an answer from stale or unavailable sources."
            )

        lines = [
            f"[KU KNOWLEDGE CONTEXT] Topic: {topic}",
            f"Freshness gate: {self.threshold:.2f} | "
            f"{len(admitted)} documents admitted | decay-validated\n",
        ]

        for i, doc in enumerate(admitted, 1):
            retraction_note = ""
            lines.append(
                f"[DOC {i}] {doc.title}\n"
                f"  Platform:   {doc.platform} | Published: {doc.publication_date}\n"
                f"  Decay:      {doc.decay_score:.3f} ({doc.label}) | "
                f"Age: {doc.age_days}d | {doc.days_until_stale}d until stale\n"
                f"  Quality:    {doc.quality_score:.1f} | "
                f"Peer-reviewed: {doc.peer_reviewed}{retraction_note}\n"
                f"  URL:        {doc.url}\n"
                f"  Summary:    {doc.summary}\n"
            )

        lines.append(
            "[END KU CONTEXT] All documents above have passed temporal decay validation. "
            "Synthesise only from these sources."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Report printer (for terminal / demo)
    # ------------------------------------------------------------------
    def print_report(self, result: GateResult) -> None:
        from rich.console import Console
        from rich.table   import Table
        from rich.panel   import Panel

        console = Console()
        console.print(Panel(
            f"[bold]Topic:[/bold] {result.topic}\n"
            f"[bold]Threshold:[/bold] {result.threshold:.2f}\n"
            f"[bold]Retrieved:[/bold] {result.total_retrieved} docs\n"
            f"[bold green]Admitted:[/bold green] {result.admitted_count} docs\n"
            f"[bold red]Blocked:[/bold red]  {result.blocked_count} docs\n"
            f"[bold green]Tokens saved:[/bold green] ~{result.tokens_saved:,} "
            f"(never sent to LLM)",
            title="[bold]Decay gate result[/bold]",
        ))

        table = Table(show_lines=True)
        table.add_column("Decision",  justify="center")
        table.add_column("Title",                         max_width=40)
        table.add_column("Platform",  justify="center")
        table.add_column("Decay",     justify="right")
        table.add_column("Label",     justify="center")
        table.add_column("Age (d)",   justify="right")
        table.add_column("Est. tokens", justify="right")

        for doc in result.admitted_docs:
            table.add_row(
                "[green]✓ PASS[/green]", doc.title[:40], doc.platform,
                f"{doc.decay_score:.3f}", doc.label,
                str(doc.age_days), str(doc.estimated_tokens),
            )
        for doc in result.blocked_docs:
            table.add_row(
                "[red]⊘ GATE[/red]", doc.title[:40], doc.platform,
                f"[red]{doc.decay_score:.3f}[/red]", f"[red]{doc.label}[/red]",
                str(doc.age_days), str(doc.estimated_tokens),
            )

        console.print(table)

        vel = result.knowledge_velocity
        if vel:
            console.print(
                f"\n[bold]Knowledge velocity:[/bold] {vel.get('velocity_label', '?').upper()} — "
                f"{vel.get('interpretation', '')}"
            )

        conflicts = result.conflict_detection
        if conflicts.get("conflicts_found"):
            console.print(
                f"\n[bold yellow]⚠ Conflicts detected:[/bold yellow] "
                f"{conflicts['conflicts_found']} pair(s)"
            )


# ------------------------------------------------------------------
# Smoke test — python decay_gate.py
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from ku_client import KUClient

    client = KUClient()
    gate   = DecayGate(threshold=0.40)

    print("Calling KU API...")
    ku_resp = client.discover(
        topic="best practices deploying LLM agents enterprise",
        difficulty=3,
        max_results=10,
    )

    result = gate.apply(ku_resp, topic="LLM agent deployment")
    gate.print_report(result)
    print("\n--- Context string (first 600 chars) ---")
    print(result.context_text[:600])