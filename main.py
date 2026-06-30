"""
Deep-Think-Agent 项目入口

使用 asyncio 运行 LangGraph 智能体，生成分析报告。

使用示例：
    python main.py "2026年新能源汽车发展前景如何？"
    python main.py --quiet "人工智能对就业市场的影响"   # 不显示节点进度
    python main.py   # 交互式输入问题
"""

import argparse
import asyncio
import os
import sys
import time

from dotenv import load_dotenv

from agent.graph import NODE_LABELS, run_agent, run_agent_with_progress

load_dotenv()


def _check_env() -> None:
    zhipu_key = os.getenv("ZHIPUAI_API_KEY")
    serpapi_key = os.getenv("SERPAPI_API_KEY")

    if not zhipu_key or zhipu_key == "your_zhipu_api_key_here":
        print("警告：ZHIPUAI_API_KEY 未配置，请在 .env 文件中设置。")
    if not serpapi_key or serpapi_key == "your_serpapi_api_key_here":
        print("警告：SERPAPI_API_KEY 未配置，web_search 将使用 mock 数据。")


def _print_progress(step: int, node_name: str, summary: str, delta: dict) -> None:
    label = NODE_LABELS.get(node_name, node_name)
    print(f"  [{step:>2}] {label}")
    print(f"       └─ {summary}")
    if delta.get("errors") and node_name in (
        "perception_filter",
        "modeling_extract",
        "reasoning_test",
        "report_gen",
    ):
        print(f"       ⚠ {delta['errors'][-1]}")


async def main() -> None:
    _check_env()

    parser = argparse.ArgumentParser(description="Deep-Think-Agent 深思熟虑型智能体")
    parser.add_argument("query", nargs="?", help="要分析的问题")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，不显示节点进度（默认开启流式进度）",
    )
    args = parser.parse_args()

    user_query = (args.query or input("\n请输入您的问题：").strip())
    if not user_query:
        print("错误：问题不能为空。")
        sys.exit(1)

    print(f"\n问题：{user_query}\n")

    if args.quiet:
        print("正在分析...\n")
        final_state = await run_agent(user_query)
    else:
        print("=" * 60)
        print("执行进度（节点流式输出）")
        print("=" * 60)
        started = time.perf_counter()
        final_state = await run_agent_with_progress(user_query, on_progress=_print_progress)
        elapsed = time.perf_counter() - started
        print("=" * 60)
        print(f"全部节点执行完毕（耗时 {elapsed:.1f}s）\n")

    print("=" * 60)
    print("最终报告")
    print("=" * 60 + "\n")
    print(final_state.get("final_report", "(无报告内容)"))

    report_path = final_state.get("report_path")
    if report_path:
        print(f"\n报告已保存至：{report_path}")

    errors = final_state.get("errors", [])
    if errors:
        print(f"\n运行过程中产生 {len(errors)} 条错误：")
        for err in errors:
            print(f"  - {err}")


if __name__ == "__main__":
    asyncio.run(main())
