"""
感知 + 建模阶段节点联调脚本。

依次调用 perception / modeling 节点，打印每一步后的状态摘要。
"""

import asyncio
import json
from typing import Any, Callable, Dict, List

from dotenv import load_dotenv

from agent.nodes import (
    modeling_extract,
    modeling_uncertainty,
    perception_filter,
    perception_query_gen,
    perception_search,
)
from agent.state import AgentState, create_initial_state

load_dotenv()

NODES: List[tuple[str, Callable[[AgentState], Any]]] = [
    ("perception_query_gen", perception_query_gen),
    ("perception_search", perception_search),
    ("perception_filter", perception_filter),
    ("modeling_extract", modeling_extract),
    ("modeling_uncertainty", modeling_uncertainty),
]

WATCH_KEYS = [
    "search_queries",
    "raw_sources",
    "filtered_sources",
    "world_model",
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
            for item in value
        ]

    if key == "world_model":
        wm = value or {}
        return {
            "entities_count": len(wm.get("entities", [])),
            "relations_count": len(wm.get("relations", [])),
            "timeline_count": len(wm.get("timeline", [])),
            "uncertainty_map_count": len(wm.get("uncertainty_map", {})),
            "entities": wm.get("entities", [])[:3],
            "relations": wm.get("relations", [])[:3],
            "timeline": wm.get("timeline", [])[:3],
            "uncertainty_map": dict(list(wm.get("uncertainty_map", {}).items())[:3]),
        }

    return value


def _print_step(step: int, node_name: str, update: Dict[str, Any], state: AgentState) -> None:
    print("=" * 60)
    print(f"步骤 {step}: {node_name}")
    print("-" * 60)
    print("本步更新字段:", list(update.keys()) or "(无)")
    for key in WATCH_KEYS:
        if key not in update and key not in state:
            continue
        print(f"\n[{key}]")
        print(json.dumps(_summarize_value(key, state.get(key)), ensure_ascii=False, indent=2))
    print()


async def main() -> None:
    user_query = "2024年大语言模型有哪些主要发展趋势？"
    state = create_initial_state(user_query)

    print("初始 state")
    print(json.dumps({"user_query": state["user_query"], "next_step": state["next_step"]}, ensure_ascii=False, indent=2))
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
