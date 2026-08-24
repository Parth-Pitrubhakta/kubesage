"""Graph nodes.

Each node takes state and returns a partial state update. Nodes never raise
into the graph — a failure records itself in `errors` and returns whatever
partial result it has, so one dead dependency degrades the answer instead of
killing the request.
"""

from __future__ import annotations

import logging

from langgraph.types import interrupt

from app.cache.tiered import Tier, cache
from app.config import get_settings
from app.graph.state import GraphState
from app.llm import complete, complete_structured
from app.models import ReflectResult
from app.retrieval.grader import grade_relevance, web_fallback
from app.retrieval.hybrid import hybrid_search
from app.retrieval.hyde import generate_hypotheses
from app.retrieval.rerank import rerank
from app.security.injection import spotlight
from app.text2sql.generate import (
    execute_sql,
    format_rows,
    generate_sql,
)
from app.text2sql.validate import SQLValidationError, validate_sql

log = logging.getLogger(__name__)
settings = get_settings()


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

_ROUTER_SYSTEM = """Classify a Kubernetes operator's question into one route.

rag    — conceptual or how-to questions answered by documentation
         ("why do pods get OOMKilled", "how do I configure an HPA")
sql    — questions about the current state of a specific cluster, needing
         live data ("which pods restarted most today", "how many nodes are
         NotReady")
hybrid — needs both live data and explanation
         ("why is my checkout pod crashing and how do I fix it")

Answer with exactly one word: rag, sql, or hybrid."""


async def route_intent(state: GraphState) -> GraphState:
    query = state["query"]

    cached = await cache.get(Tier.INTENT, query)
    if cached is not None:
        return {"route": cached, "cache_hit": True}

    try:
        raw = await complete(query, system=_ROUTER_SYSTEM, max_tokens=8)
        route = raw.strip().lower()
        if route not in {"rag", "sql", "hybrid"}:
            log.warning("router returned %r, defaulting to rag", route)
            route = "rag"
    except Exception as exc:
        log.error("routing failed, defaulting to rag: %s", exc)
        return {"route": "rag", "errors": [f"router: {exc}"]}

    await cache.set(Tier.INTENT, query, route)
    return {"route": route}


# --------------------------------------------------------------------------
# RAG branch
# --------------------------------------------------------------------------


async def expand_query(state: GraphState) -> GraphState:
    hypotheses = await generate_hypotheses(state["query"])
    return {"hypotheses": hypotheses}


async def retrieve(state: GraphState) -> GraphState:
    docs = await hybrid_search(state["query"], state.get("hypotheses"))
    return {"documents": docs}


async def rerank_documents(state: GraphState) -> GraphState:
    docs = await rerank(state["query"], state.get("documents", []))
    return {"documents": docs}


async def grade_documents(state: GraphState) -> GraphState:
    score = await grade_relevance(state["query"], state.get("documents", []))
    return {"crag_score": score}


async def search_web(state: GraphState) -> GraphState:
    docs = await web_fallback(state["query"])
    existing = state.get("documents", [])
    # Keep local docs too — the web results supplement rather than replace.
    return {"documents": existing + docs, "used_web_fallback": True}


async def build_context(state: GraphState) -> GraphState:
    docs = state.get("documents", [])
    chunks = [d["text"] for d in docs[: settings.rerank_top_k]]
    return {"context": spotlight(chunks)}


# --------------------------------------------------------------------------
# SQL branch
# --------------------------------------------------------------------------


async def write_sql(state: GraphState) -> GraphState:
    try:
        raw_sql = await generate_sql(state["query"])
        safe_sql = validate_sql(raw_sql)
    except SQLValidationError as exc:
        log.warning("generated SQL rejected by validator: %s", exc)
        return {"generated_sql": "", "errors": [f"sql_validation: {exc}"]}
    except Exception as exc:
        log.error("SQL generation failed: %s", exc)
        return {"generated_sql": "", "errors": [f"sql_generation: {exc}"]}

    return {"generated_sql": safe_sql}


async def await_approval(state: GraphState) -> GraphState:
    """Suspend the graph until a human approves the SQL.

    interrupt() raises out of the node; LangGraph persists state to the
    checkpointer and returns control to the caller. Resuming with a Command
    re-enters this node, and interrupt() returns the supplied value.
    """
    sql = state.get("generated_sql")
    if not sql:
        return {"sql_approved": False, "sql_rejection_reason": "no SQL generated"}

    decision = interrupt({
        "type": "sql_approval",
        "sql": sql,
        "question": state["query"],
    })

    approved = bool(decision.get("approved"))
    return {
        "sql_approved": approved,
        "sql_rejection_reason": None if approved else decision.get("reason"),
    }


async def run_sql(state: GraphState) -> GraphState:
    try:
        rows = await execute_sql(state["generated_sql"])
    except Exception as exc:
        log.error("SQL execution failed: %s", exc)
        return {"sql_rows": [], "errors": [f"sql_execution: {exc}"]}

    return {"sql_rows": rows, "sql_context": format_rows(rows)}


# --------------------------------------------------------------------------
# Generation and reflection
# --------------------------------------------------------------------------

_ANSWER_SYSTEM = """You are a Kubernetes operations assistant.

Answer from the supplied context only. If the context does not contain the
answer, say so plainly — do not fill the gap from general knowledge.

Be concise and concrete. Prefer specific commands and manifest fields over
general advice. Where the context supports a claim, ground it there."""


async def generate_answer(state: GraphState) -> GraphState:
    # A re-entry means Self-RAG sent us back. The initial generation is not
    # a regeneration, so the counter is owned here rather than in reflect() —
    # incrementing on every reflection fires the retry cap one attempt early.
    is_retry = bool(state.get("answer"))
    attempts = state.get("regeneration_count", 0) + (1 if is_retry else 0)

    parts = []
    if state.get("context"):
        parts.append(state["context"])
    if state.get("sql_context"):
        parts.append(f"<query_results>\n{state['sql_context']}\n</query_results>")

    if not parts:
        return {
            "answer": "I couldn't retrieve anything relevant to answer that.",
            "self_rag_score": 0.0,
            "regeneration_count": attempts,
        }

    prompt = f"{chr(10).join(parts)}\n\nQuestion: {state['query']}"
    if is_retry:
        prompt += (
            "\n\nA previous attempt was judged insufficiently grounded in the "
            "context. Stay closer to what the context actually states, and say "
            "plainly where it falls short."
        )

    try:
        answer = await complete(prompt, system=_ANSWER_SYSTEM, max_tokens=1024)
    except Exception as exc:
        log.error("answer generation failed: %s", exc)
        return {"answer": "", "errors": [f"generation: {exc}"]}

    return {"answer": answer, "regeneration_count": attempts}


_REFLECT_SYSTEM = """Score whether an answer is grounded in its context.

1.0 — every claim traces to the context
0.5 — partially grounded, some unsupported claims
0.0 — largely unsupported by the context

Judge grounding, not helpfulness. An answer that correctly says the context
is insufficient scores high."""


async def reflect(state: GraphState) -> GraphState:
    answer = state.get("answer", "")
    if not answer:
        return {"self_rag_score": 0.0}

    context = (state.get("context", "") + state.get("sql_context", ""))[:6000]
    prompt = f"Context:\n{context}\n\nAnswer:\n{answer}"

    try:
        result: ReflectResult = await complete_structured(
            prompt, schema=ReflectResult, system=_REFLECT_SYSTEM
        )
    except Exception as exc:
        log.error("reflection failed, accepting answer: %s", exc)
        return {"self_rag_score": 1.0}

    log.info("Self-RAG score %.2f", result.score)
    return {"self_rag_score": result.score}
