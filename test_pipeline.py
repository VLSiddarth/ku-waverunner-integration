"""
test_pipeline.py
End-to-end verification of the KU + Waverunner integration.
Run before sending to Jon to confirm every layer is working.

  python test_pipeline.py
"""

import sys
import os
from rich.console import Console
from rich.panel   import Panel

console = Console()

def section(title: str):
    console.print(f"\n[bold]{title}[/bold]")
    console.print("─" * 60)

def ok(msg):  console.print(f"  [green]✓[/green] {msg}")
def fail(msg): console.print(f"  [red]✗ FAIL:[/red] {msg}")
def warn(msg): console.print(f"  [yellow]⚠[/yellow] {msg}")


def test_env():
    section("Test 1 — Environment variables")
    required = ["KU_API_KEY", "KU_BASE_URL", "GEMINI_API_KEY", "DECAY_THRESHOLD"]
    all_ok = True
    for var in required:
        val = os.getenv(var, "")
        if val:
            masked = val[:8] + "..." if len(val) > 8 else val
            ok(f"{var} = {masked}")
        else:
            fail(f"{var} not set")
            all_ok = False
    return all_ok


def test_ku_health():
    section("Test 2 — KU API health check")
    from ku_client import KUClient
    try:
        client = KUClient()
        health = client.health()
        assert health.get("status") == "healthy", f"status={health.get('status')}"
        ok(f"status={health['status']} version={health.get('version')} redis={health.get('redis')}")
        return True
    except Exception as exc:
        fail(str(exc))
        return False


def test_ku_discover():
    section("Test 3 — KU /v1/discover (live call)")
    from ku_client import KUClient
    try:
        client = KUClient()
        resp   = client.discover(
            topic="FDA clinical trial adverse event reporting",
            difficulty=3,
            max_results=5,
        )
        total = resp.get("total_found", 0)
        assert total > 0, "No documents returned"
        ok(f"total_found={total} cache_hit={resp.get('cache_hit')} "
           f"processing={resp.get('processing_time_ms', 0):.0f}ms")
        ok(f"credits_remaining={resp.get('credits_remaining')}")

        vel = resp.get("knowledge_velocity", {})
        ok(f"knowledge_velocity={vel.get('velocity_label')} "
           f"median_age={vel.get('median_age_days')}d")

        decay = resp.get("decay_scores", {})
        ok(f"{len(decay)} decay scores returned")
        return resp
    except Exception as exc:
        fail(str(exc))
        return None


def test_decay_gate(ku_resp):
    section("Test 4 — Decay gate filter")
    from decay_gate import DecayGate
    try:
        gate   = DecayGate(threshold=0.40)
        result = gate.apply(ku_resp, topic="FDA clinical trial adverse event reporting")

        ok(f"admitted={result.admitted_count} blocked={result.blocked_count}")
        ok(f"tokens_saved=~{result.tokens_saved:,}")
        ok(f"context_text length={len(result.context_text)} chars")

        assert len(result.context_text) > 10, "context_text is empty"
        assert result.admitted_count + result.blocked_count == result.total_retrieved

        if result.conflict_detection.get("conflicts_found"):
            warn(f"Conflicts detected: {result.conflict_detection['conflicts_found']}")
        else:
            ok("No source conflicts detected")

        return result
    except Exception as exc:
        fail(str(exc))
        return None


def test_gemini(gate_result):
    section("Test 5 — Gemini synthesis over gated context")
    from waverunner_agent import WaverunnerAgent
    try:
        agent = WaverunnerAgent(verbose=False)
        resp  = agent.run(
            query="What are the current best practices for adverse event reporting in clinical trials?",
            verbose=False,
        )
        if resp.error:
            fail(resp.error)
            return False

        assert len(resp.answer) > 50, "Gemini response too short"
        ok(f"Gemini responded ({len(resp.answer)} chars) in {resp.latency_ms:.0f}ms")
        ok(f"Model: {resp.model_used}")
        ok(f"Tokens saved: ~{resp.tokens_saved:,}")
        console.print(f"\n  [dim]Response preview:[/dim]")
        console.print(f"  [italic]{resp.answer[:300]}...[/italic]")
        return True
    except Exception as exc:
        fail(str(exc))
        return False


def main():
    console.print(Panel(
        "[bold]KU + Waverunner integration — end-to-end test suite[/bold]\n"
        "Tests: env → KU health → KU discover → decay gate → Gemini synthesis",
        title="Test runner",
    ))

    from dotenv import load_dotenv
    load_dotenv()

    results = []

    results.append(test_env())
    results.append(test_ku_health())

    ku_resp = test_ku_discover()
    results.append(ku_resp is not None)

    gate_result = None
    if ku_resp:
        gate_result = test_decay_gate(ku_resp)
        results.append(gate_result is not None)

    results.append(test_gemini(gate_result))

    section("Summary")
    passed = sum(1 for r in results if r)
    total  = len(results)
    color  = "green" if passed == total else "yellow" if passed > 0 else "red"
    console.print(
        f"  [{color}]{passed}/{total} tests passed[/{color}] — "
        + ("All good! Ready to send to Jon." if passed == total else "Fix failures above before demo.")
    )

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()