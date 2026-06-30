# Deep-Think-Agent

基于 LangGraph 和智谱 AI 的深思熟虑型智能体，用于复杂分析与决策。

## 项目结构

```
Deep-Think-aAgent/
├── agent/                # 核心模块
│   └── __init__.py
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
| `SERPAPI_API_KEY` | SerpAPI 搜索密钥 |

## 运行项目

```bash
python main.py
```

程序将加载环境变量并打印初始化信息。

## 技术栈

- **LangGraph** — Agent 编排框架
- **LangChain Core** — LLM 调用抽象层
- **智谱 AI** — 大语言模型服务
- **SerpAPI** — 搜索引擎 API
