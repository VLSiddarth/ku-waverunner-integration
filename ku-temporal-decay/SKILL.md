---
name: ku-temporal-decay
description: "Validates retrieved documents for temporal freshness before they enter agent context. Use when agent retrieves documents from any source and needs to prevent stale context from reaching the LLM."
---

# Knowledge Universe Temporal Decay Skill

## When to activate this skill
Activate whenever the agent retrieves documents from a vector database, search API, or any external source before passing context to the LLM.

## Instructions

1. Take the list of retrieved documents
2. POST to https://api.knowledgeuniverse.tech/v1/discover with the topic
3. Check decay_score for each source — block anything above 0.4
4. Check knowledge_velocity — if HYPERSONIC, reduce cache TTL to 24h
5. Check conflict_pairs — if conflicts found, flag to user before synthesis
6. Pass only admitted documents to the LLM context window

## Output signals
- decay_score: 0.0 (fresh) to 1.0 (fully stale)
- knowledge_velocity: frozen / slow / moderate / fast / hypersonic
- max_decay_detected: highest decay score in the retrieved set
- conflict_pairs: contradicting sources detected

## Authentication
Inject KU_API_KEY from credential store as X-API-Key header.