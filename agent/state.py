"""
Deep-Think-Agent 状态定义模块

使用 TypedDict 定义 Agent 运行时的全局状态结构，
并提供工厂函数创建带有默认值的初始状态。
"""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """Agent 运行时的全局状态。

    字段说明：
        user_query:        用户输入的原始查询
        search_queries:    拆解后的搜索查询列表
        raw_sources:       搜索引擎返回的原始结果
        filtered_sources:  经过筛选与去重后的可信来源
        world_model:       世界模型，包含实体、关系、时间线与不确定性
        hypotheses:        提出的假设列表
        test_results:      假设验证结果
        selected_option:   最终选定的方案
        decision_rationale: 决策理由
        final_report:      最终生成的报告
        report_format:     报告输出格式，默认 "markdown"
        iteration:         当前迭代轮次
        max_iterations:    最大迭代轮次
        next_step:         下一步要执行的动作名称
        errors:            运行过程中产生的错误信息
    """

    user_query: str
    search_queries: List[str]
    raw_sources: List[Dict[str, Any]]
    filtered_sources: List[Dict[str, Any]]
    world_model: Dict[str, Any]          # entities, relations, timeline, uncertainty_map
    hypotheses: List[Dict[str, Any]]
    test_results: List[Dict[str, Any]]
    selected_option: Optional[Dict[str, Any]]
    decision_rationale: str
    final_report: str
    report_format: str                   # 默认 "markdown"
    iteration: int
    max_iterations: int
    next_step: str
    errors: List[str]


def create_initial_state(user_query: str) -> AgentState:
    """创建带有默认值的初始状态。

    Args:
        user_query: 用户输入的原始查询文本。

    Returns:
        填充了默认值的 AgentState 字典。
    """
    return AgentState(
        user_query=user_query,
        search_queries=[],
        raw_sources=[],
        filtered_sources=[],
        world_model={
            "entities": [],
            "relations": [],
            "timeline": [],
            "uncertainty_map": {},
        },
        hypotheses=[],
        test_results=[],
        selected_option=None,
        decision_rationale="",
        final_report="",
        report_format="markdown",
        iteration=0,
        max_iterations=5,
        next_step="search",       # 默认从搜索开始
        errors=[],
    )
