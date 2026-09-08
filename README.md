# KubeSage

**Enterprise Advanced RAG for Kubernetes Operations**

A production-grade Retrieval-Augmented Generation system that answers Kubernetes operational questions by combining semantic search over documentation with live queries against an operations database — orchestrated as a LangGraph state machine, hardened with a nine-layer guardrail pipeline, and deployed on AWS ECS Fargate.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11-blue">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-state%20machine-purple">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-async-009688">
  <img alt="AWS" src="https://img.shields.io/badge/AWS-ECS%20Fargate-orange">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## Why this exists

An SRE debugging a cluster at 2 a.m. has two kinds of questions:

- *"Why does my pod keep getting OOMKilled?"* — answered by **documentation**.
- *"Which pods in prod restarted more than five times today?"* — answered by the **ops database**.

Most RAG systems only handle the first. KubeSage routes each query to the right pipeline, and for database questions it generates SQL, then **pauses for human approval before executing anything**.

The second design goal was honesty about retrieval failure. Naive RAG confidently answers from irrelevant chunks. KubeSage grades its own retrieved context, falls back to web search when relevance drops below threshold, and reflects on its own generated answer — regenerating when confidence is low.

---

## Architecture

```
User ──▶ FastAPI ──▶ [ 9-Layer Input Security ] ──▶ LangGraph State Machine
                                                            │
                                                    ┌───────┴────────┐
                                                Intent Router
                                                    │            │
                                            RAG Pipeline    Text2SQL Pipeline
                                                    │            │
                                            HyDE expansion   Generate SQL
                                            Hybrid retrieval Validate (SELECT-only)
                                            RRF fusion       ⏸ interrupt() → human
                                            Cross-encoder    Execute
                                            CRAG grader      Format rows
                                              │ (rel < 0.7)
                                              └─▶ Web fallback
                                                    │            │
                                                    └─────┬──────┘
                                                          ▼
                                              LLM Answer Generation
                                                          │
                                              Self-RAG Reflect (score < 0.8 → regen, max 2)
                                                          ▼
                                        [ Output Security ] ──▶ Response
```

Every expensive LLM and database call is wrapped by a five-tier Redis cache with per-tier TTLs.

The compiled graph, rendered by LangGraph itself:

```
                                      +-----------+                                  
                                      | __start__ |                                  
                                      +-----------+                                  
                                            *                                        
                                            *                                        
                                            *                                        
                                    +--------------+                                 
                                    | route_intent |                                 
                                   .+--------------+....                             
                              .....                     .....                        
                           ...                               ....                    
                        ...                                      ....                
            +--------------+                                         ...             
            | expand_query |                                           .             
            +--------------+                                           .             
                    *                                                  .             
                    *                                                  .             
                    *                                                  .             
              +----------+                                             .             
              | retrieve |                                             .             
              +----------+                                             .             
                    *                                                  .             
                    *                                                  .             
                    *                                                  .             
          +------------------+                                         .             
          | rerank_documents |                                         .             
          +------------------+                                         .             
                    *                                                  .             
                    *                                                  .             
                    *                                                  .             
          +-----------------+                                          .             
          | grade_documents |                                          .             
          +-----------------+                                          .             
             ..           ..                                           .             
           ..               ..                                         .             
         ..                   ..                                       .             
+------------+                  ..                                     .             
| search_web |                ..                                       .             
+------------+              ..                                         .             
             **           ..                                           .             
               **       ..                                             .             
                 **   ..                                               .             
            +---------------+                                          .             
            | build_context |....                                      .             
            +---------------+    .........                             .             
                    .                     ..........                   .             
                    .                               .........          .             
                    .                                        .....     .             
                    .                                           +-----------+        
                    .                                           | write_sql |        
                    .                                           +-----------+        
                    .                                                  *             
                    .                                                  *             
                    .                                                  *             
                    .                                         +----------------+     
                    .                                         | await_approval |     
                    .                                         +----------------+     
                    .                                           ...         ...      
                    .                                          .               .     
                    .                                        ..                 ...  
                    ...                              +---------+                .... 
                       ...                           | run_sql |          ......     
                          .....                      +---------+    ......           
                               ...                 **        .......                 
                                  .....          **    ......                        
                                       ..      **  ....                              
                                   +-----------------+                               
                                   | generate_answer |                               
                                   +-----------------+                               
                                            .                                        
                                            .                                        
                                            .                                        
                                       +---------+                                   
                                       | reflect |                                   
                                       +---------+                                   
                                            .                                        
                                            .                                        
                                            .                                        
                                       +---------+                                   
                                       | __end__ |                                   
                                       +---------+
```

| Layer | Technology | Role |
|---|---|---|
| API | FastAPI + Pydantic | Async REST, OpenAPI docs, schema validation |
| Orchestration | LangGraph | State machine, conditional edges, checkpointing |
| Vector store | Qdrant | Dense + sparse hybrid collection |
| Relational | PostgreSQL 16 | Ops data, LangGraph checkpoints |
| Cache | Redis | Five-tier response and embedding cache |
| Reranking | Cross-encoder | Precision reranking after fusion |
| Web fallback | Tavily | CRAG low-relevance branch |
| LLM | OpenAI GPT-4o | Generation, SQL, grading |
| Deploy | Docker + AWS ECS Fargate | Containerized serverless |
| CI/CD | GitHub Actions | Test → build → push → deploy |

---

## Retrieval pipeline

The retrieval path was built incrementally, and each technique earns its place:

**HyDE (Hypothetical Document Embeddings)** — generates three hypothetical answers to the query and embeds those instead of the raw question. Operational questions are phrased very differently from documentation prose, and embedding a hypothetical *answer* closes that vocabulary gap.

**Hybrid retrieval** — dense vectors capture semantic similarity; sparse BM25 vectors catch exact identifiers like `CrashLoopBackOff` or `ImagePullBackOff` that dense embeddings blur together.

**Reciprocal Rank Fusion (k=60)** — merges the two ranked lists without needing score normalization across incompatible scales.

**Cross-encoder reranking** — the bi-encoder retrieval stage optimizes recall; the cross-encoder scores query-document pairs jointly for precision on the final top-k.

**CRAG grading** — a grader scores retrieved context relevance. Below 0.7, the system falls back to web search rather than answering from weak context.

**Self-RAG reflection** — the generated answer is scored against retrieved context. Below 0.8, regenerate, capped at two attempts to bound latency.

---

## Security pipeline

Nine layers, applied on the request side and again on the response side.

**Request side:** Pydantic schema and regex injection-pattern matching (L1), JWT bearer authentication (L4a), per-user rate limiting at 20 req/min (L4b), token budget enforcement at 100k/day/user (L6), input restructuring and tiktoken truncation (L5), `llm-guard` prompting and toxicity scan (L2), content moderation with PII redaction (L7a).

**Response side:** output moderation and PII redaction (L7b), Pydantic schema validation with automatic LLM retry on schema failure (L9).

**Spotlighting (L8)** wraps retrieved chunks in XML delimiters before they reach the generation prompt, so the model can distinguish instructions from retrieved content — a defense against indirect prompt injection through the document corpus.

> **Known limitation.** SQL validation uses a SELECT-only blocklist. Blocklists are weaker than allowlists for injection defense; a production hardening step would be an allowlisted query grammar plus a read-only database role, so that a bypass fails at the database boundary rather than at the application layer.

---

## Human-in-the-loop

Generated SQL never executes automatically. LangGraph's `interrupt()` suspends the graph at the approval node, persists state to the Postgres checkpointer, and returns the proposed query to the caller. On approval the graph resumes from the checkpoint; on rejection it terminates and returns the rejection reason.

Because state is checkpointed rather than held in memory, approval can arrive minutes later, from a different process, or after a container restart.

---

## Caching

| Tier | TTL | Rationale |
|---|---|---|
| Embedding | 7 days | Query embeddings are deterministic per model version |
| Intent router | 24 hours | Intent classification is stable for repeated phrasings |
| SQL generation | 24 hours | Schema changes are infrequent |
| SQL result | 15 minutes | Ops data moves; short TTL bounds staleness |
| RAG answer | 1 hour | Balances freshness against generation cost |

Cache keys are SHA-256 hashes of the normalized query plus tier identifier.

---

## Getting started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI API key
- Qdrant instance (Cloud free tier or local container)
- Tavily API key (optional — disables web fallback if absent)

### Local setup

```bash
git clone https://github.com/Parth-Pitrubhakta/kubesage.git
cd kubesage

cp .env.example .env
# Fill in OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY,
# POSTGRES_URL, REDIS_URL, JWT_SECRET, TAVILY_API_KEY

docker compose up -d          # Postgres, Redis, Qdrant

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m scripts.ingest --source data/k8s-docs/    # build the index
uvicorn app.main:app --reload
```

API docs at `http://localhost:8000/docs`.

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | yes | Generation and embeddings |
| `QDRANT_URL` / `QDRANT_API_KEY` | yes | Vector store connection |
| `POSTGRES_URL` | yes | Ops DB and LangGraph checkpoints |
| `REDIS_URL` | yes | Cache backend |
| `JWT_SECRET` | yes | Bearer token signing |
| `TAVILY_API_KEY` | no | Web fallback; disabled if unset |
| `CRAG_THRESHOLD` | no | Relevance cutoff, default 0.7 |
| `SELF_RAG_THRESHOLD` | no | Reflection cutoff, default 0.8 |

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/query` | Submit a query; returns answer or pending-approval state |
| `POST` | `/api/v1/approve/{thread_id}` | Approve or reject generated SQL |
| `GET` | `/api/v1/health` | Liveness and dependency checks |
| `GET` | `/api/v1/metrics` | Cache hit rates, latency percentiles |

**Example**

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "Why do my pods keep getting OOMKilled?"}'
```

```json
{
  "answer": "Pods are OOMKilled when a container exceeds its memory limit...",
  "route": "rag",
  "sources": [
    {"title": "Managing Resources for Containers", "score": 0.89}
  ],
  "cache_hit": false,
  "self_rag_score": 0.91,
  "latency_ms": 3140
}
```

---

## Evaluation

A golden dataset of 40 labelled queries spanning documentation lookups, database questions, and hybrid cases.

| Metric | Score |
|---|---|
| Retrieval precision@5 | not yet measured |
| Answer accuracy (LLM judge) | not yet measured |
| Intent routing accuracy | not yet measured |
| Cache hit rate (warm) | not yet measured |
| p95 latency, cold | not yet measured |
| p95 latency, warm | not yet measured |

Run `python -m eval.run` against a live instance to populate these. The harness writes `eval/last_run.json` and prints a summary; the numbers here should be copied from a real run rather than estimated.

```bash
python -m eval.run --dataset eval/golden_set.json
```

Ablations comparing baseline RAG against each added technique live in [`eval/RESULTS.md`](eval/RESULTS.md).

---

## Testing

```bash
pytest                                # 65 tests
pytest --cov=app --cov-report=term    # with coverage
```

Unit tests cover the RRF implementation, SQL validation (including stacked statements, CTE-wrapped writes, and system-catalog access), injection heuristics, PII redaction, and each conditional edge in isolation.

Integration tests in `tests/test_graph_integration.py` run the compiled graph with only the LLM calls mocked. That distinction matters more than it sounds: the Self-RAG regeneration test was wrong twice before it was right. The first version mocked `generate_answer`, so the counter logic never executed. The second mocked `reflect`, so a bug living inside `reflect` was unreachable — and that is exactly where the original off-by-one was. Both versions passed against known-broken code.

The bug itself: `reflect` incremented the retry counter on the first reflection, before any regeneration had occurred, so the cap fired one attempt early and only one retry ever ran. `generate_answer` now owns the counter and increments only on re-entry. The fix is verified by mutation — reintroduce the increment and `test_regeneration_bounded_at_retry_cap` fails with `expected 3 generations, got 2`.

The lesson generalises: a test that mocks the component containing the bug will pass no matter how broken that component is.

---
## Deployment

GitHub Actions runs lint → test → build → push to ECR → deploy to ECS Fargate on every push to `main`. Failing tests block the deploy.

```bash
docker build -t kubesage:latest .
docker run -p 8000:8000 --env-file .env kubesage:latest
```

---

## Project layout

```
kubesage/
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── graph/               # LangGraph nodes, edges, state
│   ├── retrieval/           # HyDE, hybrid search, RRF, reranking
│   ├── security/            # 9-layer guardrail pipeline
│   ├── cache/               # 5-tier Redis wrapper
│   └── text2sql/            # Generation, validation, approval
├── scripts/ingest.py        # Corpus ingestion
├── eval/                    # Golden set and harness
├── tests/
├── .github/workflows/
└── docker-compose.yml
```

---

## Roadmap

- Allowlisted SQL grammar replacing the blocklist validator
- Streaming responses via server-sent events
- Terraform modules for the AWS infrastructure
- Multi-tenant corpus isolation

---

## License

MIT — see [LICENSE](LICENSE).
#
