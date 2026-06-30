"""
Deep-Think-Agent LangGraph 工作流

构建完整的感知 → 建模 → 推理 → 决策 → 报告 流水线。
"""

from typing import AsyncIterator, Literal

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    decision_ranking,
    modeling_extract,
    modeling_uncertainty,
    perception_filter,
    perception_query_gen,
    perception_search,
    reasoning_hypothesis_gen,
    reasoning_test,
    reflection,
    report_format,
    report_gen,
)
from agent.state import AgentState, create_initial_state


def _route_after_reflection(
    state: AgentState,
) -> Literal["back_to_search", "back_to_reasoning", "proceed"]:
    next_step = state.get("next_step", "proceed")
    if next_step == "back_to_search":
        return "back_to_search"
    if next_step == "back_to_reasoning":
        return "back_to_reasoning"
    return "proceed"


def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("perception_query_gen", perception_query_gen)
    graph.add_node("perception_search", perception_search)
    graph.add_node("perception_filter", perception_filter)
    graph.add_node("modeling_extract", modeling_extract)
    graph.add_node("modeling_uncertainty", modeling_uncertainty)
    graph.add_node("reasoning_hypothesis_gen", reasoning_hypothesis_gen)
    graph.add_node("reasoning_test", reasoning_test)
    graph.add_node("reflection", reflection)
    graph.add_node("decision_ranking", decision_ranking)
    graph.add_node("report_gen", report_gen)
    graph.add_node("report_format", report_format)

    graph.add_edge(START, "perception_query_gen")
    graph.add_edge("perception_query_gen", "perception_search")
    graph.add_edge("perception_search", "perception_filter")
    graph.add_edge("perception_filter", "modeling_extract")
    graph.add_edge("modeling_extract", "modeling_uncertainty")
    graph.add_edge("modeling_uncertainty", "reasoning_hypothesis_gen")
    graph.add_edge("reasoning_hypothesis_gen", "reasoning_test")
    graph.add_edge("reasoning_test", "reflection")

    graph.add_conditional_edges(
        "reflection",
        _route_after_reflection,
        {
            "back_to_search": "perception_query_gen",
            "back_to_reasoning": "reasoning_hypothesis_gen",
            "proceed": "decision_ranking",
        },
    )

    graph.add_edge("decision_ranking", "report_gen")
    graph.add_edge("report_gen", "report_format")
    graph.add_edge("report_format", END)

    return graph


app = _build_graph().compile()


def make_initial_state(user_query: str) -> AgentState:
    """创建带默认 max_iterations=3 的初始状态。"""
    state = create_initial_state(user_query)
    state["max_iterations"] = 3
    return state


async def run_agent(user_query: str) -> dict:
    """运行完整 Agent 工作流，返回最终状态。"""
    return await app.ainvoke(make_initial_state(user_query))


async def run_agent_stream(user_query: str) -> AsyncIterator[dict]:
    """逐节点流式输出状态更新，便于调试。"""
    async for update in app.astream(make_initial_state(user_query), stream_mode="updates"):
        yield update
