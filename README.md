# Deep-Think-Agent

基于 LangGraph 和智谱 AI 的深思熟虑型智能体，用于复杂分析、多源信息综合与决策报告生成。

## 特性

- **感知 → 建模 → 推理 → 决策 → 报告** 五阶段流水线
- 联网搜索（SerpAPI）+ 网页抓取，自动筛选可靠来源
- 世界模型构建（实体、关系、时间线、不确定性标注）
- 多假设生成、证据验证与反思迭代（最多 3 轮）
- **流式节点进度输出**，实时查看执行到哪一步
- 自动生成 Markdown 报告并保存至 `output/` 目录

## 项目结构

```
Deep-Think-aAgent/
├── agent/
│   ├── state.py          # AgentState 状态定义
│   ├── tools.py          # 联网搜索、网页抓取
│   ├── model.py          # 智谱 GLM 模型封装（call_glm）
│   ├── prompts.py        # 各阶段 Prompt 模板
│   ├── nodes.py          # LangGraph 节点函数
│   └── graph.py          # 工作流编排、流式运行入口
├── tests/
│   ├── conftest.py       # pytest fixtures（mock 环境）
│   ├── mock_data.py      # 五类测试场景预设数据
│   ├── test_agent.py     # Agent 集成测试（pytest + mock）
│   └── test_nodes.py     # 节点联调脚本（真实 API）
├── output/               # 生成的 Markdown 报告（git 忽略）
├── main.py               # CLI 入口
├── requirements.txt
└── README.md
```

## 安装

```bash
git clone <仓库地址>
cd Deep-Think-aAgent

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install pytest-asyncio   # 运行测试需要
```

## 环境变量

```bash
cp .env.example .env
```

| 变量名 | 说明 |
|---|---|
| `ZHIPUAI_API_KEY` | 智谱 AI API 密钥 |
| `SERPAPI_API_KEY` | SerpAPI 搜索密钥（[获取地址](https://serpapi.com/manage-api-key)） |

> 未配置 `SERPAPI_API_KEY` 时，`web_search` 自动回退 mock 数据，便于离线开发。

## 运行

```bash
# 传入问题（默认显示节点流式进度）
python main.py "2026年下半年买黄金还是比特币？"

# 静默模式，不打印节点进度
python main.py --quiet "日本15天深度游，预算2万"

# 交互式输入
python main.py
```

### 流式进度输出

默认会实时打印每个节点的执行情况：

```
============================================================
执行进度（节点流式输出）
============================================================
  [ 1] 感知 · 生成搜索查询
       └─ 生成 3 条搜索词
  [ 2] 感知 · 联网搜索与抓取
       └─ 获取 15 条原始来源
  [ 3] 感知 · 筛选可靠来源
       └─ 筛选保留 5 条来源
  ...
  [ 8] 推理 · 反思与迭代决策
       └─ 第 1 轮反思 → 进入决策
  [11] 报告 · 保存文件
       └─ 已保存 output/report_xxx.md
============================================================
全部节点执行完毕（耗时 45.2s）
```

若反思判定信息不足，会显示 `→ 回到搜索` 或 `→ 回到推理` 并继续循环，直到进入决策或达到 `max_iterations`（默认 3）。

运行结束后终端打印完整报告，同时在 `output/` 生成 `report_{问题摘要}_{时间戳}.md`。

## 工作流

```
START
  → 感知：查询生成 → 搜索 → 来源筛选
  → 建模：实体抽取 → 不确定性标注
  → 推理：假设生成 → 假设验证 → 反思
        ├─ back_to_search    → 回到查询生成
        ├─ back_to_reasoning → 回到假设生成
        └─ proceed           → 决策排序 → 报告生成 → 保存文件 → END
```

## 核心模块

| 模块 | 说明 |
|---|---|
| `agent/graph.py` | LangGraph 图编译；`run_agent()` / `run_agent_with_progress()` |
| `agent/nodes.py` | 11 个异步节点，每节点返回部分状态更新 |
| `agent/prompts.py` | 查询生成、筛选、抽取、假设、反思、决策、报告等 Prompt |
| `agent/model.py` | `call_glm()`，模型 `glm-4.6v-flashx`，支持 JSON 模式与重试 |
| `agent/tools.py` | `web_search()`（SerpAPI）、`fetch_page_content()` |
| `agent/state.py` | `AgentState` + `create_initial_state()` |

### 编程调用

```python
import asyncio
from agent.graph import run_agent, run_agent_with_progress

# 静默运行
state = asyncio.run(run_agent("巴黎奥运会哪一年举办？"))

# 带进度回调
def on_progress(step, node, summary, delta):
    print(f"[{step}] {summary}")

state = asyncio.run(
    run_agent_with_progress("你的问题", on_progress=on_progress)
)
print(state["final_report"])
print(state["report_path"])
```

## 测试

测试使用 **pytest + mock**，mock 了 `web_search`、`fetch_page_content`、`call_glm`，**无需真实 API Key**。

```bash
pytest tests/ -v
pytest tests/test_agent.py -v   # 仅集成测试
```

覆盖场景：投资决策、行程规划、矛盾信息、知识空白、简单事实。

`tests/test_nodes.py` 为真实 API 联调脚本（需配置 `.env`）：

```bash
python tests/test_nodes.py
```

## 技术栈

- **LangGraph** — Agent 编排与条件循环
- **智谱 AI（GLM-4.6v-flashx）** — 大语言模型
- **SerpAPI** — Google 搜索引擎 API
- **BeautifulSoup4 / Requests** — 网页解析
- **pytest / pytest-asyncio** — 测试
