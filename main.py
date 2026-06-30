"""
Deep-Think-Agent 项目入口

使用 asyncio 运行 LangGraph 智能体，生成分析报告。

使用示例：
    python main.py "2026年新能源汽车发展前景如何？"
    python main.py --stream "人工智能对就业市场的影响"
    python main.py   # 交互式输入问题
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

from agent.graph import run_agent, run_agent_stream

load_dotenv()


def _check_env() -> None:
    zhipu_key = os.getenv("ZHIPUAI_API_KEY")
    serpapi_key = os.getenv("SERPAPI_API_KEY")

    if not zhipu_key or zhipu_key == "your_zhipu_api_key_here":
        print("警告：ZHIPUAI_API_KEY 未配置，请在 .env 文件中设置。")
    if not serpapi_key or serpapi_key == "your_serpapi_api_key_here":
        print("警告：SERPAPI_API_KEY 未配置，web_search 将使用 mock 数据。")


async def _run_with_stream(user_query: str) -> dict:
    final_state: dict = {"user_query": user_query}

    print(f"\n问题：{user_query}\n")
    print("=" * 60)

    async for update in run_agent_stream(user_query):
        for node_name, delta in update.items():
            print(f"[节点完成] {node_name} -> 更新字段: {list(delta.keys())}")
            final_state.update(delta)

    print("=" * 60)
    return final_state


async def main() -> None:
    _check_env()

    parser = argparse.ArgumentParser(description="Deep-Think-Agent 深思熟虑型智能体")
    parser.add_argument("query", nargs="?", help="要分析的问题")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="流式输出每个节点的执行进度",
    )
    args = parser.parse_args()

    user_query = (args.query or input("\n请输入您的问题：").strip())
    if not user_query:
        print("错误：问题不能为空。")
        sys.exit(1)

    if args.stream:
        final_state = await _run_with_stream(user_query)
    else:
        print(f"\n正在分析问题：{user_query}\n")
        final_state = await run_agent(user_query)

    print("\n" + "=" * 60)
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
