"""End-to-end graph traversal with every I/O boundary mocked.

test_graph.py covers the edge functions in isolation. These run the actual
compiled graph, which is the only way to catch wiring faults — an edge
pointing at the wrong node, or state that doesn't survive a node boundary.

The Self-RAG test here caught a real off-by-one: reflect() was incrementing
the retry counter on the first reflection, before any regeneration had
happened, so the cap fired one attempt early. Isolated edge tests could not
have caught it, because the edge function was correct — the state feeding it
was wrong.
"""

from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.graph.builder import build_graph
from app.models import ReflectResult

SQL = "SELECT name FROM pods WHERE restart_count > 5 LIMIT 500"

DOC = {"id": "d1", "text": "context text", "title": "Doc", "score": 1.0}


def rag_patches(**overrides):
    """Mock every node up to generation. Overrides replace individual nodes."""
    defaults = {
        "route_intent": AsyncMock(return_value={"route": "rag"}),
        "expand_query": AsyncMock(return_value={"hypotheses": ["h"]}),
        "retrieve": AsyncMock(return_value={"documents": [DOC]}),
        "rerank_documents": AsyncMock(return_value={}),
        "grade_documents": AsyncMock(return_value={"crag_score": 0.9}),
        "build_context": AsyncMock(return_value={"context": "ctx"}),
        "generate_answer": AsyncMock(
            return_value={"answer": "an answer", "regeneration_count": 0}
        ),
        "reflect": AsyncMock(return_value={"self_rag_score": 0.95}),
    }
    defaults.update(overrides)
    return [patch(f"app.graph.nodes.{k}", v) for k, v in defaults.items()]


async def invoke(patches, initial, thread="t"):
    for p in patches:
        p.start()
    try:
        graph = build_graph(checkpointer=MemorySaver())
        return await graph.ainvoke(
            initial, config={"configurable": {"thread_id": thread}}
        )
    finally:
        for p in patches:
            p.stop()


class TestSelfRagCycle:
    async def test_regeneration_bounded_at_retry_cap(self):
        """Both generate_answer and reflect must run for real.

        This test has been wrong twice. First it mocked generate_answer, so
        the counter logic never executed. Then it mocked reflect, so a bug
        *inside* reflect was unreachable — and the original off-by-one lived
        in reflect. Only the two LLM calls are mocked now; every line that
        touches regeneration_count executes.
        """
        gen_calls = {"n": 0}

        async def fake_complete(prompt, **kwargs):
            gen_calls["n"] += 1
            return f"answer v{gen_calls['n']}"

        async def fake_structured(prompt, schema, **kwargs):
            # Always ungrounded, so the cycle runs until the cap stops it.
            return ReflectResult(score=0.2, grounded=False, reasoning="test")

        patches = [
            patch("app.graph.nodes.route_intent",
                  AsyncMock(return_value={"route": "rag"})),
            patch("app.graph.nodes.expand_query",
                  AsyncMock(return_value={"hypotheses": ["h"]})),
            patch("app.graph.nodes.retrieve",
                  AsyncMock(return_value={"documents": [DOC]})),
            patch("app.graph.nodes.rerank_documents", AsyncMock(return_value={})),
            patch("app.graph.nodes.grade_documents",
                  AsyncMock(return_value={"crag_score": 0.9})),
            patch("app.graph.nodes.build_context",
                  AsyncMock(return_value={"context": "ctx"})),
            # generate_answer and reflect deliberately NOT mocked.
            patch("app.graph.nodes.complete", fake_complete),
            patch("app.graph.nodes.complete_structured", fake_structured),
        ]

        result = await invoke(
            patches, {"query": "q", "regeneration_count": 0}, thread="selfrag"
        )

        # Initial generation plus self_rag_max_retries regenerations.
        assert gen_calls["n"] == 3, (
            f"expected 3 generations (1 initial + 2 retries), got {gen_calls['n']}"
        )
        assert result["regeneration_count"] == 2

    async def test_retry_prompt_mentions_grounding(self):
        """Regenerations must tell the model why the last attempt failed,
        otherwise the retry is just a resample at temperature 0."""
        prompts: list[str] = []

        async def capture(prompt, **kwargs):
            prompts.append(prompt)
            return "answer"

        patches = [
            patch("app.graph.nodes.route_intent",
                  AsyncMock(return_value={"route": "rag"})),
            patch("app.graph.nodes.expand_query",
                  AsyncMock(return_value={"hypotheses": ["h"]})),
            patch("app.graph.nodes.retrieve",
                  AsyncMock(return_value={"documents": [DOC]})),
            patch("app.graph.nodes.rerank_documents", AsyncMock(return_value={})),
            patch("app.graph.nodes.grade_documents",
                  AsyncMock(return_value={"crag_score": 0.9})),
            patch("app.graph.nodes.build_context",
                  AsyncMock(return_value={"context": "ctx"})),
            patch("app.graph.nodes.reflect",
                  AsyncMock(return_value={"self_rag_score": 0.2})),
            patch("app.graph.nodes.complete", capture),
        ]

        await invoke(patches, {"query": "q", "regeneration_count": 0},
                     thread="retry-prompt")

        assert "grounded" not in prompts[0].lower()
        assert "grounded" in prompts[1].lower()

    async def test_grounded_answer_generates_once(self):
        gen = AsyncMock(return_value={"answer": "a", "regeneration_count": 0})
        await invoke(
            rag_patches(generate_answer=gen), {"query": "q", "regeneration_count": 0}
        )
        assert gen.await_count == 1


class TestCragBranch:
    async def test_low_relevance_reaches_for_web(self):
        web = AsyncMock(return_value={"documents": [DOC], "used_web_fallback": True})
        result = await invoke(
            rag_patches(
                grade_documents=AsyncMock(return_value={"crag_score": 0.3}),
                search_web=web,
            ),
            {"query": "q"},
            thread="crag-low",
        )
        assert web.await_count == 1
        assert result["used_web_fallback"] is True

    async def test_high_relevance_skips_web(self):
        web = AsyncMock(return_value={})
        await invoke(
            rag_patches(
                grade_documents=AsyncMock(return_value={"crag_score": 0.95}),
                search_web=web,
            ),
            {"query": "q"},
            thread="crag-high",
        )
        assert web.await_count == 0


class TestApprovalGate:
    def _sql_patches(self, executed):
        async def fake_run_sql(state):
            executed.append(state["generated_sql"])
            return {"sql_rows": [{"name": "p"}], "sql_context": "name\np"}

        return [
            patch("app.graph.nodes.route_intent",
                  AsyncMock(return_value={"route": "sql"})),
            patch("app.graph.nodes.write_sql",
                  AsyncMock(return_value={"generated_sql": SQL})),
            patch("app.graph.nodes.run_sql", fake_run_sql),
            patch("app.graph.nodes.generate_answer",
                  AsyncMock(return_value={"answer": "a", "regeneration_count": 0})),
            patch("app.graph.nodes.reflect",
                  AsyncMock(return_value={"self_rag_score": 0.95})),
        ]

    async def test_suspends_without_executing(self):
        executed: list[str] = []
        patches = self._sql_patches(executed)
        for p in patches:
            p.start()
        try:
            graph = build_graph(checkpointer=MemorySaver())
            result = await graph.ainvoke(
                {"query": "q"}, config={"configurable": {"thread_id": "gate-1"}}
            )
            assert "__interrupt__" in result
            assert result["__interrupt__"][0].value["sql"] == SQL
            # The point of the gate: nothing ran.
            assert executed == []
        finally:
            for p in patches:
                p.stop()

    async def test_approval_resumes_and_executes(self):
        executed: list[str] = []
        patches = self._sql_patches(executed)
        for p in patches:
            p.start()
        try:
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "gate-2"}}
            await graph.ainvoke({"query": "q"}, config=config)
            result = await graph.ainvoke(
                Command(resume={"approved": True}), config=config
            )
            assert executed == [SQL]
            assert result["answer"] == "a"
        finally:
            for p in patches:
                p.stop()

    async def test_rejection_never_executes(self):
        executed: list[str] = []
        patches = self._sql_patches(executed)
        for p in patches:
            p.start()
        try:
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "gate-3"}}
            await graph.ainvoke({"query": "q"}, config=config)
            result = await graph.ainvoke(
                Command(resume={"approved": False, "reason": "too broad"}),
                config=config,
            )
            assert executed == []
            assert result["sql_rejection_reason"] == "too broad"
        finally:
            for p in patches:
                p.stop()


class TestTopology:
    def test_graph_compiles(self):
        graph = build_graph()
        nodes = set(graph.get_graph().nodes)
        for expected in ["route_intent", "await_approval", "reflect", "search_web"]:
            assert expected in nodes

    def test_no_orphan_nodes(self):
        graph = build_graph().get_graph()
        reachable = {e.target for e in graph.edges} | {"__start__"}
        orphans = set(graph.nodes) - reachable
        assert not orphans, f"unreachable nodes: {orphans}"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Guard: any unmocked outbound call fails loudly rather than hanging."""
    monkeypatch.setattr(
        "app.llm._client", None, raising=False
    )
