"""
Deep-Think-Agent LangGraph 工作流

构建完整的感知 → 建模 → 推理 → 决策 → 报告 流水线。
"""

from typing import Any, AsyncIterator, Callable, Dict, Literal, Optional

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

# 节点中文名称，用于流式进度展示
NODE_LABELS: Dict[str, str] = {
    "perception_query_gen": "感知 · 生成搜索查询",
    "perception_search": "感知 · 联网搜索与抓取",
    "perception_filter": "感知 · 筛选可靠来源",
    "modeling_extract": "建模 · 抽取实体与关系",
    "modeling_uncertainty": "建模 · 标注不确定性",
    "reasoning_hypothesis_gen": "推理 · 生成候选假设",
    "reasoning_test": "推理 · 验证假设证据",
    "reflection": "推理 · 反思与迭代决策",
    "decision_ranking": "决策 · 方案排序",
    "report_gen": "报告 · 生成 Markdown",
    "report_format": "报告 · 保存文件",
}

ProgressCallback = Callable[[int, str, str, Dict[str, Any]], None]


def _route_after_reflection(
    state: AgentState,
) -> Literal["back_to_search", "back_to_reasoning", "proceed"]:
    next_step = state.get("next_step", "proceed")
    if next_step == "back_to_search":
        return "back_to_search"
    if next_step == "back_to_reasoning":
        return "back_to_reasoning"
    return "proceed"


def summarize_node_update(node_name: str, delta: Dict[str, Any]) -> str:
    """将节点输出压缩为一行进度摘要。"""
    if node_name == "perception_query_gen":
        queries = delta.get("search_queries", [])
        return f"生成 {len(queries)} 条搜索词"
    if node_name == "perception_search":
        return f"获取 {len(delta.get('raw_sources', []))} 条原始来源"
    if node_name == "perception_filter":
        return f"筛选保留 {len(delta.get('filtered_sources', []))} 条来源"
    if node_name == "modeling_extract":
        wm = delta.get("world_model", {})
        return (
            f"实体 {len(wm.get('entities', []))} · "
            f"关系 {len(wm.get('relations', []))} · "
            f"事件 {len(wm.get('timeline', []))}"
        )
    if node_name == "modeling_uncertainty":
        wm = delta.get("world_model", {})
        return f"不确定性标注 {len(wm.get('uncertainty_map', {}))} 项"
    if node_name == "reasoning_hypothesis_gen":
        return f"提出 {len(delta.get('hypotheses', []))} 个候选假设"
    if node_name == "reasoning_test":
        return f"完成 {len(delta.get('test_results', []))} 条假设验证"
    if node_name == "reflection":
        iteration = delta.get("iteration", "?")
        next_step = delta.get("next_step", "?")
        route_hint = {
            "back_to_search": "→ 回到搜索",
            "back_to_reasoning": "→ 回到推理",
            "proceed": "→ 进入决策",
        }.get(str(next_step), f"→ {next_step}")
        return f"第 {iteration} 轮反思 {route_hint}"
    if node_name == "decision_ranking":
        option = delta.get("selected_option") or {}
        hypothesis = option.get("hypothesis", "")[:40]
        return f"选定方案：{hypothesis or '(无)'}"
    if node_name == "report_gen":
        length = len(delta.get("final_report", ""))
        return f"报告 {length} 字"
    if node_name == "report_format":
        path = delta.get("report_path", "")
        return f"已保存 {path}" if path else "格式 markdown"
    if delta.get("errors"):
        return f"完成（{len(delta['errors'])} 条错误记录）"
    return "完成"


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
    """逐节点流式输出状态更新（LangGraph updates 模式）。"""
    async for update in app.astream(make_initial_state(user_query), stream_mode="updates"):
        yield update


async def run_agent_with_progress(
    user_query: str,
    on_progress: Optional[ProgressCallback] = None,
) -> dict:
    """流式运行 Agent，逐节点回调进度并返回最终状态。"""
    final_state: dict = dict(make_initial_state(user_query))
    step = 0

    async for update in run_agent_stream(user_query):
        for node_name, delta in update.items():
            step += 1
            label = NODE_LABELS.get(node_name, node_name)
            summary = summarize_node_update(node_name, delta)
            if on_progress:
                on_progress(step, node_name, summary, delta)
            final_state.update(delta)

    return final_state
