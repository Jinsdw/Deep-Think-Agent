"""
Deep-Think-Agent 节点模块

实现 Agent 各阶段的 LangGraph 节点函数。
每个节点接收 AgentState，返回部分状态更新字典。
"""

import asyncio
import json
import re
from typing import Any, Dict, List

from agent.model import call_glm
from agent.prompts import (
    DECISION_RANKING_PROMPT,
    ENTITY_EXTRACT_PROMPT,
    FILTER_PROMPT,
    HYPOTHESIS_GEN_PROMPT,
    HYPOTHESIS_TEST_PROMPT,
    QUERY_GEN_PROMPT,
    REFLECTION_PROMPT,
    REPORT_GEN_PROMPT,
    UNCERTAINTY_PROMPT,
)
from agent.state import AgentState
from agent.tools import fetch_page_content, web_search

TOP_RESULTS_PER_QUERY = 5
MIN_FILTERED_SOURCES = 3

NEXT_ACTION_TO_STEP = {
    "back_to_search": "search",
    "back_to_reasoning": "reasoning",
    "proceed": "proceed",
}


def _append_errors(state: AgentState, message: str) -> List[str]:
    errors = list(state.get("errors", []))
    errors.append(message)
    return errors


def _parse_json(text: str) -> Any:
    """解析模型返回的 JSON，兼容 markdown 代码块包裹。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _serialize_sources(sources: List[Dict[str, Any]]) -> str:
    """将来源列表序列化为供模型阅读的文本。"""
    payload = []
    for index, source in enumerate(sources):
        payload.append({
            "index": index,
            "title": source.get("title", ""),
            "url": source.get("url", ""),
            "snippet": source.get("snippet", ""),
            "content": (source.get("content") or "")[:1500],
        })
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _default_world_model() -> Dict[str, Any]:
    return {
        "entities": [],
        "relations": [],
        "timeline": [],
        "uncertainty_map": {},
    }


def _default_uncertainty_map(world_model: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """标注失败时为 world_model 条目生成默认低不确定性映射。"""
    uncertainty_map: Dict[str, Dict[str, str]] = {}

    for entity in world_model.get("entities", []):
        key = entity.get("id") or entity.get("name", "")
        if key:
            uncertainty_map[str(key)] = {
                "level": "low",
                "reason": "不确定性标注失败，默认为低不确定性",
            }

    for index, relation in enumerate(world_model.get("relations", [])):
        key = relation.get("description") or f"relation_{index}"
        uncertainty_map[str(key)] = {
            "level": "low",
            "reason": "不确定性标注失败，默认为低不确定性",
        }

    for index, event in enumerate(world_model.get("timeline", [])):
        key = event.get("event") or f"timeline_{index}"
        uncertainty_map[str(key)] = {
            "level": "low",
            "reason": "不确定性标注失败，默认为低不确定性",
        }

    return uncertainty_map


async def _fetch_source_content(item: Dict[str, Any]) -> Dict[str, Any]:
    content = await fetch_page_content(item.get("url", ""))
    return {
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "snippet": item.get("snippet", ""),
        "content": content,
    }


async def perception_query_gen(state: AgentState) -> dict:
    """将用户问题拆解为 3~5 条搜索查询。"""
    try:
        response = await call_glm(
            messages=[
                {"role": "system", "content": QUERY_GEN_PROMPT},
                {"role": "user", "content": f"用户问题：{state['user_query']}"},
            ],
            response_format="json",
        )
        queries = _parse_json(response)
        if not isinstance(queries, list):
            raise ValueError("搜索查询解析结果不是 JSON 数组")

        search_queries = [str(item).strip() for item in queries if str(item).strip()]
        return {"search_queries": search_queries}
    except Exception as exc:
        return {
            "search_queries": [],
            "errors": _append_errors(state, f"perception_query_gen 失败：{exc}"),
        }


async def perception_search(state: AgentState) -> dict:
    """并发搜索并抓取每个查询排名前 5 的结果页面。"""
    raw_sources: List[Dict[str, Any]] = []
    errors = list(state.get("errors", []))
    search_queries = state.get("search_queries", [])

    if not search_queries:
        return {
            "raw_sources": [],
            "errors": _append_errors(state, "perception_search：无可用搜索查询"),
        }

    for query in search_queries:
        try:
            results = await web_search(query)
            top_results = results[:TOP_RESULTS_PER_QUERY]
            fetched = await asyncio.gather(
                *[_fetch_source_content(item) for item in top_results],
                return_exceptions=True,
            )
            for index, item in enumerate(fetched):
                if isinstance(item, Exception):
                    errors.append(
                        f"perception_search 抓取失败（查询={query!r}，序号={index}）：{item}"
                    )
                    continue
                raw_sources.append(item)
        except Exception as exc:
            errors.append(f"perception_search 搜索失败（查询={query!r}）：{exc}")

    update: dict = {"raw_sources": raw_sources}
    if len(errors) > len(state.get("errors", [])):
        update["errors"] = errors
    return update


async def perception_filter(state: AgentState) -> dict:
    """评估并筛选最相关可靠的来源。"""
    raw_sources = state.get("raw_sources", [])
    if not raw_sources:
        return {
            "filtered_sources": [],
            "errors": _append_errors(state, "perception_filter：raw_sources 为空"),
        }

    try:
        response = await call_glm(
            messages=[
                {"role": "system", "content": FILTER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{state['user_query']}\n\n"
                        f"来源列表：\n{_serialize_sources(raw_sources)}"
                    ),
                },
            ],
            response_format="json",
        )
        indices = _parse_json(response)
        if not isinstance(indices, list):
            raise ValueError("筛选结果不是 JSON 数组")

        filtered_sources = [
            raw_sources[index]
            for index in indices
            if isinstance(index, int) and 0 <= index < len(raw_sources)
        ]

        update: dict = {"filtered_sources": filtered_sources}
        if len(filtered_sources) < MIN_FILTERED_SOURCES:
            update["errors"] = _append_errors(
                state,
                f"perception_filter：筛选后来源不足 {MIN_FILTERED_SOURCES} 条（实际 {len(filtered_sources)} 条）",
            )
        return update
    except Exception as exc:
        return {
            "filtered_sources": [],
            "errors": _append_errors(state, f"perception_filter 失败：{exc}"),
        }


async def modeling_extract(state: AgentState) -> dict:
    """从筛选来源中抽取实体、关系与时间线，构建世界模型。"""
    filtered_sources = state.get("filtered_sources", [])
    current_world_model = state.get("world_model") or _default_world_model()

    if not filtered_sources:
        return {
            "world_model": current_world_model,
            "errors": _append_errors(state, "modeling_extract：filtered_sources 为空"),
        }

    source_text = "\n\n".join(
        (
            f"[源 {index}] 标题：{source.get('title', '')}\n"
            f"URL：{source.get('url', '')}\n"
            f"摘要：{source.get('snippet', '')}\n"
            f"正文：{(source.get('content') or '')[:2000]}"
        )
        for index, source in enumerate(filtered_sources)
    )

    try:
        response = await call_glm(
            messages=[
                {"role": "system", "content": ENTITY_EXTRACT_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{state['user_query']}\n\n"
                        f"来源内容：\n{source_text}"
                    ),
                },
            ],
            response_format="json",
        )
        parsed = _parse_json(response)
        if not isinstance(parsed, dict):
            raise ValueError("世界模型解析结果不是 JSON 对象")

        world_model = {
            "entities": parsed.get("entities", []),
            "relations": parsed.get("relations", []),
            "timeline": parsed.get("timeline", []),
            "uncertainty_map": current_world_model.get("uncertainty_map", {}),
        }
        return {"world_model": world_model}
    except Exception as exc:
        return {
            "world_model": current_world_model,
            "errors": _append_errors(state, f"modeling_extract 失败：{exc}"),
        }


async def modeling_uncertainty(state: AgentState) -> dict:
    """为世界模型中的事实标注不确定性映射。"""
    current_world_model = state.get("world_model") or _default_world_model()
    filtered_sources = state.get("filtered_sources", [])

    if not current_world_model.get("entities") and not current_world_model.get("relations"):
        return {
            "world_model": {
                **current_world_model,
                "uncertainty_map": _default_uncertainty_map(current_world_model),
            },
            "errors": _append_errors(state, "modeling_uncertainty：world_model 为空，使用默认不确定性"),
        }

    try:
        response = await call_glm(
            messages=[
                {"role": "system", "content": UNCERTAINTY_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{state['user_query']}\n\n"
                        f"世界模型：\n{json.dumps(current_world_model, ensure_ascii=False, indent=2)}\n\n"
                        f"来源摘要：\n{_serialize_sources(filtered_sources)}"
                    ),
                },
            ],
            response_format="json",
        )
        parsed = _parse_json(response)
        if not isinstance(parsed, dict):
            raise ValueError("不确定性解析结果不是 JSON 对象")

        uncertainty_map = parsed.get("uncertainty_map", {})
        if not isinstance(uncertainty_map, dict):
            raise ValueError("uncertainty_map 不是 JSON 对象")

        return {
            "world_model": {
                **current_world_model,
                "uncertainty_map": uncertainty_map,
            }
        }
    except Exception as exc:
        return {
            "world_model": {
                **current_world_model,
                "uncertainty_map": _default_uncertainty_map(current_world_model),
            },
            "errors": _append_errors(state, f"modeling_uncertainty 失败：{exc}"),
        }


async def reasoning_hypothesis_gen(state: AgentState) -> dict:
    """基于世界模型生成候选假设列表。"""
    world_model = state.get("world_model") or _default_world_model()

    try:
        response = await call_glm(
            messages=[
                {"role": "system", "content": HYPOTHESIS_GEN_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{state['user_query']}\n\n"
                        f"世界模型：\n{json.dumps(world_model, ensure_ascii=False, indent=2)}"
                    ),
                },
            ],
            response_format="json",
        )
        parsed = _parse_json(response)
        if not isinstance(parsed, list):
            raise ValueError("假设解析结果不是 JSON 数组")

        return {"hypotheses": parsed}
    except Exception as exc:
        return {
            "hypotheses": [],
            "errors": _append_errors(state, f"reasoning_hypothesis_gen 失败：{exc}"),
        }


async def reasoning_test(state: AgentState) -> dict:
    """逐条验证假设，收集证据评估结果。"""
    hypotheses = state.get("hypotheses", [])
    world_model = state.get("world_model") or _default_world_model()
    filtered_sources = state.get("filtered_sources", [])

    if not hypotheses:
        return {
            "test_results": [],
            "errors": _append_errors(state, "reasoning_test：hypotheses 为空"),
        }

    test_results: List[Dict[str, Any]] = []
    errors = list(state.get("errors", []))

    for index, hypothesis in enumerate(hypotheses):
        try:
            response = await call_glm(
                messages=[
                    {"role": "system", "content": HYPOTHESIS_TEST_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"用户问题：{state['user_query']}\n\n"
                            f"假设索引：{index}\n"
                            f"假设内容：\n{json.dumps(hypothesis, ensure_ascii=False, indent=2)}\n\n"
                            f"世界模型：\n{json.dumps(world_model, ensure_ascii=False, indent=2)}\n\n"
                            f"来源摘要：\n{_serialize_sources(filtered_sources)}"
                        ),
                    },
                ],
                response_format="json",
            )
            parsed = _parse_json(response)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        item.setdefault("hypothesis_index", index)
                        test_results.append(item)
            elif isinstance(parsed, dict):
                parsed.setdefault("hypothesis_index", index)
                test_results.append(parsed)
            else:
                raise ValueError("假设测试结果格式无效")
        except Exception as exc:
            errors.append(f"reasoning_test 失败（假设索引={index}）：{exc}")
            test_results.append({
                "hypothesis_index": index,
                "supporting_evidence": [],
                "opposing_evidence": [],
                "overall_score": 0.0,
            })

    update: dict = {"test_results": test_results}
    if len(errors) > len(state.get("errors", [])):
        update["errors"] = errors
    return update


async def reflection(state: AgentState) -> dict:
    """评估推理是否充分，决定下一步动作并递增迭代计数。"""
    iteration = state.get("iteration", 0) + 1
    max_iterations = state.get("max_iterations", 5)
    hypotheses = state.get("hypotheses", [])
    test_results = state.get("test_results", [])

    try:
        response = await call_glm(
            messages=[
                {"role": "system", "content": REFLECTION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{state['user_query']}\n\n"
                        f"当前迭代：{iteration} / {max_iterations}\n\n"
                        f"假设列表：\n{json.dumps(hypotheses, ensure_ascii=False, indent=2)}\n\n"
                        f"测试结果：\n{json.dumps(test_results, ensure_ascii=False, indent=2)}"
                    ),
                },
            ],
            response_format="json",
        )
        parsed = _parse_json(response)
        if not isinstance(parsed, dict):
            raise ValueError("反思结果不是 JSON 对象")

        next_action = parsed.get("next_action", "proceed")
        if next_action not in NEXT_ACTION_TO_STEP:
            raise ValueError(f"无效的 next_action：{next_action}")

        next_step = NEXT_ACTION_TO_STEP[next_action]
    except Exception as exc:
        update = {
            "iteration": iteration,
            "next_step": "proceed",
            "errors": _append_errors(state, f"reflection 失败：{exc}"),
        }
        if iteration >= max_iterations:
            update["next_step"] = "proceed"
        return update

    if iteration >= max_iterations:
        next_step = "proceed"

    return {"iteration": iteration, "next_step": next_step}


async def decision_ranking(state: AgentState) -> dict:
    """对候选方案排序并选出最优解。"""
    hypotheses = state.get("hypotheses", [])
    test_results = state.get("test_results", [])
    world_model = state.get("world_model") or _default_world_model()

    if not hypotheses:
        return {
            "selected_option": None,
            "decision_rationale": "",
            "errors": _append_errors(state, "decision_ranking：hypotheses 为空"),
        }

    try:
        response = await call_glm(
            messages=[
                {"role": "system", "content": DECISION_RANKING_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{state['user_query']}\n\n"
                        f"假设列表：\n{json.dumps(hypotheses, ensure_ascii=False, indent=2)}\n\n"
                        f"测试结果：\n{json.dumps(test_results, ensure_ascii=False, indent=2)}\n\n"
                        f"世界模型：\n{json.dumps(world_model, ensure_ascii=False, indent=2)}"
                    ),
                },
            ],
            response_format="json",
        )
        parsed = _parse_json(response)
        if not isinstance(parsed, dict):
            raise ValueError("决策排序结果不是 JSON 对象")

        ranking = parsed.get("ranking", [])
        selected_index = parsed.get("selected_index", 0)
        if not isinstance(selected_index, int) or not (0 <= selected_index < len(hypotheses)):
            selected_index = 0

        selected_option: Dict[str, Any] = {
            **hypotheses[selected_index],
            "selected_index": selected_index,
            "ranking": ranking,
        }

        decision_rationale = ""
        if ranking and isinstance(ranking[0], dict):
            decision_rationale = str(ranking[0].get("reason", ""))
        if not decision_rationale:
            decision_rationale = hypotheses[selected_index].get("reasoning", "")

        return {
            "selected_option": selected_option,
            "decision_rationale": decision_rationale,
        }
    except Exception as exc:
        fallback_option = {**hypotheses[0], "selected_index": 0} if hypotheses else None
        return {
            "selected_option": fallback_option,
            "decision_rationale": hypotheses[0].get("reasoning", "") if hypotheses else "",
            "errors": _append_errors(state, f"decision_ranking 失败：{exc}"),
        }


async def report_gen(state: AgentState) -> dict:
    """生成带引用的 Markdown 分析报告。"""
    try:
        response = await call_glm(
            messages=[
                {"role": "system", "content": REPORT_GEN_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{state['user_query']}\n\n"
                        f"世界模型：\n{json.dumps(state.get('world_model', {}), ensure_ascii=False, indent=2)}\n\n"
                        f"假设列表：\n{json.dumps(state.get('hypotheses', []), ensure_ascii=False, indent=2)}\n\n"
                        f"测试结果：\n{json.dumps(state.get('test_results', []), ensure_ascii=False, indent=2)}\n\n"
                        f"选定方案：\n{json.dumps(state.get('selected_option'), ensure_ascii=False, indent=2)}\n\n"
                        f"决策理由：{state.get('decision_rationale', '')}\n\n"
                        f"引用来源（filtered_sources）：\n{_serialize_sources(state.get('filtered_sources', []))}"
                    ),
                },
            ],
        )
        final_report = response.strip()
        if not final_report:
            raise ValueError("模型返回空报告")

        return {"final_report": final_report}
    except Exception as exc:
        return {
            "final_report": (
                f"# 报告生成失败\n\n"
                f"用户问题：{state.get('user_query', '')}\n\n"
                f"错误信息：{exc}"
            ),
            "errors": _append_errors(state, f"report_gen 失败：{exc}"),
        }


async def report_format(state: AgentState) -> dict:
    """设置报告输出格式为 Markdown。"""
    return {"report_format": "markdown"}
