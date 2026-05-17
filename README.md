# KU + Waverunner Integration — Image 1 Architecture

Live demonstration of the Knowledge Universe temporal decay gate integrated
with a Gemini-powered Waverunner agent pipeline.

**Author:** V.L. Siddarth — Knowledge Universe API  
**Live API:** https://vlsiddarth-knowledge-universe.hf.space  
**EAP context:** Waverunner feedback thread (Jon Matthews)

---

## What this demonstrates

This is the Image 1 architecture from our conversation:

```
Waverunner agent
    → POST /v1/discover (authenticated, KU API)
    → Temporal decay scoring (per-document decay_score + label)
    → Freshness gate (hard-blocks docs above threshold, 0 tokens consumed)
    → Gemini synthesis over validated, fresh context only
```

The key result from our clinical NLP test suite: **~50% reduction in input
token burn** on queries where a significant fraction of retrieved documents
are temporally stale. A retracted FDA protocol with cosine similarity 0.94
is gated before it consumes a single token.

---

## Setup (2 minutes)

```bash
git clone <this repo>
cd ku-waverunner-integration

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your GEMINI_API_KEY from https://aistudio.google.com
```

`.env` contents:
```
KU_API_KEY=your_injected_sandbox_key_here
KU_BASE_URL=https://vlsiddarth-knowledge-universe.hf.space
GEMINI_API_KEY=your_key_here
DECAY_THRESHOLD=0.40
```

---

## Run the demo

```bash
# Full demo (2 queries, shows gate in action)
python demo_run.py

# Custom query
python demo_run.py "FDA adverse event reporting clinical trials 2026"

# End-to-end test suite (run this first)
python test_pipeline.py
```

---

## File map

| File | Purpose |
|------|---------|
| `ku_client.py` | Authenticated KU API wrapper |
| `decay_gate.py` | Temporal decay filter — the Image 1 middleware |
| `waverunner_agent.py` | Gemini + KU integrated agent |
| `demo_run.py` | Demo script for the Waverunner team |
| `test_pipeline.py` | End-to-end test suite |

---

## The decay formula

```python
decay_score = 1 - 0.5 ^ (age_days / half_life)
```

Domain-specific half-lives (days):
- arXiv papers: 1095
- Regulatory (FDA, FINRA): 365
- GitHub repos: 180
- YouTube / tutorials: 120
- HuggingFace model cards: 90

A retracted source receives an additional +0.35 penalty on top of its
age-based score, pushing it above any reasonable gate threshold.

---

## What Image 2 would look like

The current integration (Image 1) requires the developer to route their
retrieval pipeline through KU explicitly. Image 2 — a native Waverunner
pre-context governance hook — would allow KU to register as a trusted
middleware service that intercepts documents from any source, automatically,
before they reach any agent's context window.

We have the scoring logic, velocity classification, and conflict detection
already built. We need the sandbox registration hook that Waverunner would
need to expose.

---

## Contact

V.L. Siddarth  
vlsiddarth7@gmail.com  
https://vlsiddarth-knowledge-universe.hf.space