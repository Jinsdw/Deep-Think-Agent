"""
全链路节点联调脚本。

依次调用 perception / modeling / reasoning / decision / report 节点，
打印每一步后的状态摘要。

运行方式（在项目根目录）：
    python tests/test_nodes.py
    python -m tests.test_nodes
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

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

load_dotenv(ROOT / ".env")

NodeFn = Callable[[AgentState], Awaitable[dict]]

NODES: List[tuple[str, NodeFn]] = [
    ("perception_query_gen", perception_query_gen),
    ("perception_search", perception_search),
    ("perception_filter", perception_filter),
    ("modeling_extract", modeling_extract),
    ("modeling_uncertainty", modeling_uncertainty),
    ("reasoning_hypothesis_gen", reasoning_hypothesis_gen),
    ("reasoning_test", reasoning_test),
    ("reflection", reflection),
    ("decision_ranking", decision_ranking),
    ("report_gen", report_gen),
    ("report_format", report_format),
]

WATCH_KEYS = [
    "search_queries",
    "raw_sources",
    "filtered_sources",
    "world_model",
    "hypotheses",
    "test_results",
    "iteration",
    "next_step",
    "selected_option",
    "decision_rationale",
    "final_report",
    "report_format",
    "errors",
]


def _summarize_value(key: str, value: Any) -> Any:
    if key == "search_queries":
        return value

    if key in ("raw_sources", "filtered_sources"):
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": (item.get("snippet") or "")[:80],
                "content_len": len(item.get("content") or ""),
            }
            for item in (value or [])
        ]

    if key == "world_model":
        wm = value or {}
        return {
            "entities_count": len(wm.get("entities", [])),
            "relations_count": len(wm.get("relations", [])),
            "timeline_count": len(wm.get("timeline", [])),
            "uncertainty_map_count": len(wm.get("uncertainty_map", {})),
            "entities": wm.get("entities", [])[:2],
            "relations": wm.get("relations", [])[:2],
            "timeline": wm.get("timeline", [])[:2],
            "uncertainty_map": dict(list(wm.get("uncertainty_map", {}).items())[:2]),
        }

    if key == "hypotheses":
        return [
            {
                "hypothesis": item.get("hypothesis", ""),
                "confidence": item.get("confidence"),
                "risks": (item.get("risks") or [])[:2],
            }
            for item in (value or [])
        ]

    if key == "test_results":
        return [
            {
                "hypothesis_index": item.get("hypothesis_index"),
                "overall_score": item.get("overall_score"),
                "supporting_count": len(item.get("supporting_evidence") or []),
                "opposing_count": len(item.get("opposing_evidence") or []),
            }
            for item in (value or [])
        ]

    if key == "selected_option":
        if not value:
            return None
        return {
            "hypothesis": value.get("hypothesis", ""),
            "selected_index": value.get("selected_index"),
            "confidence": value.get("confidence"),
        }

    if key == "decision_rationale":
        text = str(value or "")
        return text[:200] + ("..." if len(text) > 200 else "")

    if key == "final_report":
        text = str(value or "")
        return {
            "length": len(text),
            "preview": text[:300] + ("..." if len(text) > 300 else ""),
        }

    return value


def _print_step(step: int, node_name: str, update: Dict[str, Any], state: AgentState) -> None:
    print("=" * 60)
    print(f"步骤 {step}: {node_name}")
    print("-" * 60)
    print("本步更新字段:", list(update.keys()) or "(无)")

    keys_to_show = list(dict.fromkeys(list(update.keys()) + WATCH_KEYS))
    for key in keys_to_show:
        if key not in state:
            continue
        print(f"\n[{key}]")
        print(json.dumps(_summarize_value(key, state.get(key)), ensure_ascii=False, indent=2))
    print()


async def main() -> None:
    user_query = "2026年新能源汽车发展前景如何？"
    state = create_initial_state(user_query)

    print("初始 state")
    print(json.dumps({
        "user_query": state["user_query"],
        "next_step": state["next_step"],
        "iteration": state["iteration"],
    }, ensure_ascii=False, indent=2))
    print()

    for step, (node_name, node_fn) in enumerate(NODES, start=1):
        update = await node_fn(state)
        state.update(update)
        _print_step(step, node_name, update, state)

    print("=" * 60)
    print("全部节点执行完毕")
    print(f"errors 共 {len(state.get('errors', []))} 条:")
    for err in state.get("errors", []):
        print(f"  - {err}")


if __name__ == "__main__":
    asyncio.run(main())
