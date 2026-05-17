"""
demo_run.py

KU + Waverunner integration — live demonstration script.
This is the file you send to Jon's team.

Setup (takes 2 minutes):
  1. pip install -r requirements.txt
  2. Add your GEMINI_API_KEY to .env
  3. python demo_run.py

What this demonstrates (Image 1 architecture):
  - Waverunner agent makes an authenticated outbound call to KU /v1/discover
  - KU temporal decay engine scores every retrieved document
  - Decay gate hard-blocks documents above threshold (0 tokens consumed)
  - Gemini synthesises a response over only the fresh, validated documents
  - Full signal flow printed to terminal with latency + token saving metrics
"""

import time
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from rich.console  import Console
from rich.panel    import Panel
from rich.table    import Table
from rich.markdown import Markdown

console = Console()


DEMO_QUERIES = [
    {
        "query":       "best practices for deploying LLM agents in enterprise environments",
        "domain":      None,
        "description": "General enterprise agent deployment — tests knowledge velocity + decay across platforms",
    },
    {
        "query":       "FDA clinical trial adverse event reporting requirements 2026",
        "domain":      "clinical",
        "description": "Clinical NLP — high-stakes regulatory context, stale FDA protocols get gated",
    },
    {
        "query":       "FINRA margin requirements algorithmic trading compliance",
        "domain":      "financial",
        "description": "Financial compliance — superseded FINRA guidelines should decay and gate",
    },
]


def print_header():
    console.print(Panel(
        "[bold]Knowledge Universe API × Waverunner Agent[/bold]\n\n"
        "Image 1 integration — external middleware architecture\n\n"
        "  KU /v1/discover  →  Decay gate  →  Gemini agent  →  Response\n\n"
        "[dim]Author: V.L. Siddarth | Knowledge Universe API[/dim]\n"
        "[dim]Live API: https://vlsiddarth-knowledge-universe.hf.space[/dim]",
        title="[bold]Live demo[/bold]",
        border_style="blue",
    ))


def run_demo_query(agent, query_cfg: dict, query_num: int):
    console.print(f"\n[bold cyan]━━ Demo {query_num}: {query_cfg['description']} ━━[/bold cyan]")
    console.print(f"[dim]Query:[/dim] {query_cfg['query']}")

    t0   = time.time()
    resp = agent.run(
        query  = query_cfg["query"],
        domain = query_cfg.get("domain"),
        verbose=True,
    )
    wall = (time.time() - t0) * 1000

    if resp.error:
        console.print(f"[red]Error:[/red] {resp.error}")
        return

    # Gate summary table
    gate = resp.gate_result
    table = Table(show_lines=False, box=None)
    table.add_column("",        style="dim",   width=20)
    table.add_column("Value",   justify="right")

    table.add_row("Retrieved",       str(gate.total_retrieved))
    table.add_row("Admitted",        f"[green]{gate.admitted_count}[/green]")
    table.add_row("Gated",           f"[red]{gate.blocked_count}[/red]")
    table.add_row("Tokens saved",    f"[green]~{gate.tokens_saved:,}[/green]")
    table.add_row("Latency (wall)",  f"{wall:.0f} ms")
    table.add_row("Model",           resp.model_used)

    vel = gate.knowledge_velocity
    if vel:
        table.add_row(
            "Knowledge velocity",
            f"{vel.get('velocity_label', '?').upper()} — {vel.get('recommended_refresh_days', '?')}d refresh",
        )

    console.print(table)

    console.print(Panel(
        Markdown(resp.answer),
        title="[bold]Agent response[/bold] (synthesised over decay-gated context)",
        border_style="green",
    ))

    # Show what was blocked
    if gate.blocked_docs:
        console.print("[dim]Documents blocked by decay gate (0 tokens):[/dim]")
        for doc in gate.blocked_docs:
            console.print(
                f"  [red]⊘[/red] {doc.title[:60]} "
                f"decay={doc.decay_score:.3f} age={doc.age_days}d"
            )


def main():
    print_header()

    # Check env
    missing = [k for k in ["KU_API_KEY", "GEMINI_API_KEY"] if not os.getenv(k)]
    if missing:
        console.print(
            f"[red]Missing env vars:[/red] {', '.join(missing)}\n"
            "Add them to your .env file and re-run."
        )
        sys.exit(1)

    from waverunner_agent import WaverunnerAgent
    agent = WaverunnerAgent(
        decay_threshold = float(os.getenv("DECAY_THRESHOLD", "0.40")),
        ku_max_results  = 10,   # keep credits usage down during demo
        verbose         = False,
    )

    # Run query selection
    if len(sys.argv) > 1:
        custom_query = " ".join(sys.argv[1:])
        run_demo_query(
            agent,
            {"query": custom_query, "domain": None, "description": "Custom query"},
            query_num=1,
        )
    else:
        # Run first two demo queries (save credits)
        for i, q in enumerate(DEMO_QUERIES[:2], 1):
            run_demo_query(agent, q, query_num=i)
            if i < 2:
                console.print("\n[dim]Waiting 3s between queries...[/dim]")
                time.sleep(3)

    console.print(
        "\n[bold green]Demo complete.[/bold green] "
        "This is the Image 1 architecture running live — "
        "KU decay gate intercepting documents before the Waverunner agent context window.\n"
    )


if __name__ == "__main__":
    main()