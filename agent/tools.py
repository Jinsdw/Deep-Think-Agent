"""
Deep-Think-Agent 工具模块

提供 Agent 可调用的外部工具函数，包括：
- web_search:  通过 SerpAPI 进行网络搜索
- fetch_page_content:  抓取并解析网页可见文本
"""

import os
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup


async def web_search(query: str) -> List[Dict[str, Any]]:
    """通过 SerpAPI 搜索网络信息。

    优先使用环境变量中的 SERPAPI_API_KEY 调用真实接口；
    若未配置则返回模拟数据，方便离线开发与调试。

    Args:
        query: 搜索关键词。

    Returns:
        搜索结果列表，每项包含 title、url、snippet 字段。
    """
    api_key = os.getenv("SERPAPI_API_KEY")

    # --- 真实 API 调用 ---
    if api_key and api_key != "your_serpapi_api_key_here":
        try:
            import serpapi  # 延迟导入，避免无 key 时报错

            client = serpapi.Client(api_key=api_key)
            response = client.search(q=query, num=5, engine="google")
            organic = response.get("organic_results", [])
            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
                for item in organic
            ]
        except Exception as exc:
            print(f"SerpAPI 调用失败：{exc}，回退到 mock 数据。")

    # --- Mock 数据（无 API Key 时使用） ---
    mock_results: List[Dict[str, Any]] = [
        {
            "title": f"模拟结果 1：{query}",
            "url": "https://example.com/mock-result-1",
            "snippet": f"这是关于「{query}」的模拟搜索摘要，仅用于开发调试。",
        },
        {
            "title": f"模拟结果 2：{query} 相关资讯",
            "url": "https://example.com/mock-result-2",
            "snippet": f"深入解读「{query}」的最新动态和背景信息。",
        },
        {
            "title": f"模拟结果 3：{query} 综合分析",
            "url": "https://example.com/mock-result-3",
            "snippet": f"多方视角下的「{query}」综合分析与趋势研判。",
        },
    ]
    return mock_results


async def fetch_page_content(url: str) -> str:
    """抓取网页并提取可见文本内容。

    使用 requests 发起 HTTP GET 请求，通过 BeautifulSoup 去除
    script、style 等非内容标签，提取纯文本并截取前 3000 字符。

    Args:
        url: 目标网页 URL。

    Returns:
        提取到的网页文本；若请求失败则返回空字符串。
    """
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Deep-Think-Agent/1.0"},
            timeout=10,
        )
        response.raise_for_status()

        # 使用 html.parser 解析，无需额外安装 lxml
        soup = BeautifulSoup(response.text, "html.parser")

        # 移除 script 和 style 标签及其内容
        for tag in soup(["script", "style", "noscript", "meta", "head"]):
            tag.decompose()

        # 提取可见文本，按换行分隔
        text = soup.get_text(separator="\n", strip=True)

        # 去除多余空行，截取前 3000 字符
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)
        return clean_text[:3000]

    except (requests.RequestException, Exception):
        # 超时、连接失败、解析异常等情况统一返回空字符串
        return ""
