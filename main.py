"""
Deep-Think-Agent 项目入口文件
使用 asyncio 异步运行，加载环境变量并初始化项目。
"""

import asyncio
import os
from dotenv import load_dotenv


async def main() -> None:
    """主入口函数：加载环境变量并打印初始化信息。"""
    # 加载 .env 文件中的环境变量
    load_dotenv()

    # 验证关键环境变量是否存在
    zhipu_key = os.getenv("ZHIPUAI_API_KEY")
    serpapi_key = os.getenv("SERPAPI_API_KEY")

    if not zhipu_key or zhipu_key == "your_zhipu_api_key_here":
        print("⚠️  警告：ZHIPUAI_API_KEY 未配置，请在 .env 文件中设置。")
    if not serpapi_key or serpapi_key == "your_serpapi_api_key_here":
        print("⚠️  警告：SERPAPI_API_KEY 未配置，请在 .env 文件中设置。")

    print("项目初始化成功")


if __name__ == "__main__":
    # 使用 asyncio 运行主函数
    asyncio.run(main())
