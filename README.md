# Deep-Think-Agent

基于 LangGraph 和智谱 AI 的深思熟虑型智能体，用于复杂分析与决策。

## 项目结构

```
Deep-Think-aAgent/
├── agent/                # 核心模块
│   ├── __init__.py
│   ├── state.py          # Agent 状态定义（TypedDict）
│   └── tools.py          # 外部工具（联网搜索、网页抓取）
├── .env.example          # 环境变量模板
├── main.py               # 项目入口
├── requirements.txt      # Python 依赖
└── README.md             # 项目说明
```

## 安装步骤

1. **克隆项目**

```bash
git clone <仓库地址>
cd Deep-Think-aAgent
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

## 环境变量配置

1. 复制环境变量模板：

```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的 API Key：

| 变量名 | 说明 |
|---|---|
| `ZHIPUAI_API_KEY` | 智谱 AI API 密钥 |
| `SERPAPI_API_KEY` | SerpAPI 搜索密钥（[获取地址](https://serpapi.com/manage-api-key)） |

> 若 `SERPAPI_API_KEY` 未配置或为占位符 `your_serpapi_api_key_here`，`web_search` 将自动返回 mock 数据，便于离线开发。

## 运行项目

```bash
python main.py
```

程序将加载环境变量并打印初始化信息。

## 工具模块（`agent/tools.py`）

Agent 可调用的异步工具函数：

| 函数 | 说明 | 返回值 |
|---|---|---|
| `web_search(query)` | 通过 SerpAPI（Google 引擎）搜索网络信息 | 列表，每项含 `title`、`url`、`snippet` |
| `fetch_page_content(url)` | 抓取网页并提取可见文本（最多 3000 字符） | 字符串 |

联网搜索使用官方 [serpapi](https://github.com/serpapi/serpapi-python) Python SDK（`serpapi.Client`），无需安装旧版 `google-search-results` 包。

### 测试联网搜索

在项目根目录执行：

```bash
python -c "
import asyncio
from dotenv import load_dotenv
from agent.tools import web_search, fetch_page_content

load_dotenv()

async def test():
    results = await web_search('Python asyncio')
    for i, r in enumerate(results, 1):
        print(f'{i}. {r[\"title\"]}')
        print(f'   {r[\"url\"]}')
        print(f'   {r[\"snippet\"][:80]}...')
    text = await fetch_page_content('https://example.com')
    print('网页抓取:', text[:100] if text else '(失败)')

asyncio.run(test())
"
```

**如何判断是否真实联网：**

- 结果 URL 为 `https://example.com/mock-result-*` → 使用的是 mock 数据
- 结果 URL 为真实网站（如 `docs.python.org`）→ SerpAPI 调用成功

## 状态模块（`agent/state.py`）

使用 `TypedDict` 定义 Agent 运行时全局状态，包含：

- 用户查询与搜索词（`user_query`、`search_queries`）
- 来源与世界模型（`raw_sources`、`filtered_sources`、`world_model`）
- 假设验证与决策（`hypotheses`、`test_results`、`selected_option`）
- 报告与迭代控制（`final_report`、`iteration`、`max_iterations`）

通过 `create_initial_state()` 创建带默认值的初始状态。

## 技术栈

- **LangGraph** — Agent 编排框架
- **LangChain Core** — LLM 调用抽象层
- **智谱 AI** — 大语言模型服务
- **SerpAPI** — 搜索引擎 API（官方 Python SDK）
- **BeautifulSoup4 / Requests** — 网页抓取与解析
