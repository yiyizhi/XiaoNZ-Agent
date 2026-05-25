# XiaoNZ Agent

一个跑在飞书里的个人 AI 助手：在飞书发消息给它，它会上网搜资料、读你发的文件、记长期偏好、跨天回忆、自己学新技能，原生接 **Claude Opus 4.7**。

```text
飞书消息（文字 / 语音 / 图片 / 文件 / 视频）
        ↓
长连接 → Agent loop（Claude Opus 4.7）
        ↓
工具调用 + Skills + 长期记忆 + 跨会话召回
        ↓
回复 / 卡片 / 文件 / 图片 → 飞书
```

## 它能干什么

| 模块 | 用途 |
|------|------|
| 飞书长连接 | lark-oapi 收发飞书消息，无需公网入口 |
| Agent loop | 多轮 tool-use 推理，模型一边思考一边调工具 |
| 长期记忆 | 本地 SQLite + Markdown，Agent 自己改写、自动备份 |
| 跨会话回忆 | 每日 digest + 向量库 top-k 召回，能翻出几周前提过的事 |
| Skills | 兼容 [AgentSkills](https://agentskills.io) 标准，按需安装 |
| 23 个内置工具 | 上网、抓正文、读 PDF/Word/Excel、生图、跑 shell …… |

### 飞书集成

| 能力 | 说明 |
|------|------|
| 消息类型 | 文字 / 富文本 / 语音 / 图片 / 文件 / 视频 / 转发 全覆盖 |
| 长连接 | 无需 VPS、不用 cloudflared、不用公网 HTTPS 入口 |
| 卡片流式 | 模型一边推理一边更新飞书卡片，不是干等 |
| 主动推送 | Agent 可主动推消息 / 文件 / 图片到飞书 |
| 群聊 | 群内 @bot 触发 |

### Agent loop

| 能力 | 说明 |
|------|------|
| Tool-use | 23 个内置工具，多轮调用 + 看结果再继续 |
| Skills | 运行时按需 `install_skill` / `load_skill` 装新能力 |
| 长期记忆 | Agent 自己用工具改写 `MEMORY.md`，每次写入自动备份（保留 30 份）|
| 跨会话回忆 | 每天 cron 生成「昨日摘要」+ 向量库 top-k 语义召回 |
| 死循环熔断 | 同工具同参数重复 > 4 次自动 abort，模型刷不动 token |
| 可中断 | 用户连发新消息会取消上一轮未完成的工具循环 |

### 模型

| 能力 | 说明 |
|------|------|
| 默认 | **Claude Opus 4.7**，一行配置切 Sonnet 4.6 / Haiku 4.5 |
| 上游 | 任何 Anthropic Messages API 兼容端点（官方 / 自建网关 / 代理）|
| 流式 | `messages.stream` + per-chunk 超时，长输出不被 httpx 切断 |
| 自动压缩 | 长聊不爆 token |

### Web Search（三层级联）

`web_search` 按优先级依次尝试，前一层 ERROR 或空结果自动落下一层：

| 层级 | 后端 | 免费配额 | 优势 |
|---|---|---|---|
| 1 | **Tavily** | 1000 次/月 | LLM-tuned 重排 + AI 摘要置顶 |
| 2 | **Brave** | 2000 次/月（1 qps）| 独立索引日更，中文 query 自动 zh-hans 定位 |
| 3 | **DuckDuckGo** | 无限（无 key）| 最终兜底 |

入参支持 `freshness=day/week/month/year` 和 `topic=news`。

### 工具一览

| 类别 | 工具 |
|---|---|
| 记忆 | `update_memory` · `search_memory` · `search_memory_semantic` |
| 网络 | `web_fetch`（readability 正文提取）· `web_search`（三层级联）· `download_to_disk` |
| 多模态 | `generate_image` · `read_pdf` · `anything_to_md` · `browser_capture`（无头浏览器截图 + 抓正文）|
| 飞书 | `send_to_feishu`（主动推图 / 文件 / 卡片）|
| 文件 | `read_local_file` · `write_local_file` · `create_directory` · `list_directory` · `move_path` · `copy_path` · `delete_path` |
| Shell | `run_command`（带超时 + 输出截断）|
| Skills | `list_skills` · `load_skill` · `install_skill` · `uninstall_skill` |

## 快速开始

### 1. 准备

- Python 3.9+
- 飞书自建应用（[飞书开放平台](https://open.feishu.cn/) 创建 → 启用「机器人」能力 → 拿 `app_id` / `app_secret`）
- Anthropic API key（官方 [console.anthropic.com](https://console.anthropic.com/) 或任何兼容网关）
- *（可选）* [Tavily](https://tavily.com) / [Brave Search](https://api.search.brave.com) API key —— 不配也能跑（自动降级到 DDG）

### 2. 安装

```bash
git clone https://github.com/yiyizhi/XiaoNZ-Agent.git
cd XiaoNZ-Agent

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. 配置

```bash
cp config.yaml.example         config.yaml
cp data/memory/SOUL.md.example data/memory/SOUL.md
cp data/memory/MEMORY.md.example data/memory/MEMORY.md
```

打开 `config.yaml` 填飞书凭证 + Anthropic key（搜索 key 可选）。`SOUL.md` 按你的口味改写人设。

### 4. 启动

```bash
python -m src.main
```

前台运行 + 阻塞。生产建议用 `launchd` (macOS) / `systemd` (Linux) 守护。

## 向量召回（可选）

默认关闭。开启后 Agent 每次回复前会做 top-k 语义召回（消息 + 每日 digest + `MEMORY.md` 分块），明显提升跨会话连贯性。

> embedding 写入由后台 cron 回填，请求路径只读不写，bge-m3 抖动不会拖慢回复。

### 方案 A：Docker + TEI（推荐）

[HuggingFace text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) 专门做 embedding 推理，动态 batching，batch 任务比 Ollama 快 5~10x。

```bash
docker run -d --name tei-bge-m3 \
  -p 11434:80 \
  -v ~/.cache/huggingface:/data \
  ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
  --model-id BAAI/bge-m3
```

```yaml
embedding:
  enabled: true
  base_url: "http://127.0.0.1:11434/v1"
  model: "bge-m3"
  dim: 1024
```

### 方案 B：Ollama（无 Docker 时）

`brew install ollama` 一行装好，原生 macOS app，但单 worker 无 batching，bulk embed 慢。

```bash
brew install ollama         # 或 https://ollama.com/download
ollama pull bge-m3          # ~1.2GB
ollama serve                # 起服务在 :11434
```

`config.yaml` 同方案 A。

### 方案 C：OpenAI 官方 embeddings

```yaml
embedding:
  enabled: true
  base_url: "https://api.openai.com/v1"
  model: "text-embedding-3-large"
  dim: 3072
```

### 首次开启

跑一次把已有 SOUL / MEMORY / 历史消息全部 embed：

```bash
venv/bin/python scripts/bootstrap_memory.py
```

## License

[MIT](./LICENSE) © 2026 yiyizhi
