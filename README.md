# XiaoNZ Agent

> 轻量化的 [OpenClaw](https://github.com/openclaw/openclaw) —— 一个跑在飞书里的个人 AI 助手，原生接 **Claude Opus 4.7**。
> Single Python process · 飞书长连接 · SQLite 持久化 · 可选向量记忆。

**v0.2.1** · [MIT License](./LICENSE)

---

## 这是什么

XiaoNZ Agent 是一个长期运行的个人 Agent。它：

- 通过**飞书长连接**收发消息（无需公网 HTTPS 入口）
- 把对话、长期记忆、人设全存在本地 **SQLite + Markdown 文件**里
- 调用 **Claude Opus 4.7**（或任何 Anthropic Messages API 兼容端点）
- 每天自动生成"昨日 digest"并由后台 cron 写入向量库，让 Agent 能跨天回忆

跟完整版 OpenClaw 的区别：**不分布式、不多租户、不容器化**——一台机器一个用户一个进程，简单粗暴，核心代码 ~3k 行能看完。

---

## v0.2.1 更新一览

- **飞书消息类型全覆盖** — `post`（富文本，@/加粗/链接 会让飞书把 text 升级成 post 类型）、`audio`（优先用 `speech_to_text`）、`sticker` / `video` / `share_chat` / `merge_forward` 全部进 agent loop，不再静默 skip。参考 [OpenClaw](https://github.com/openclaw/openclaw) `extensions/feishu/src/bot-content.ts` 的 `parseMessageContent` / `parsePostContent` 实现
- **卡片 markdown 扩展** — `_preprocess_markdown` 新增 `> blockquote` → `▌` 和 `- [ ] / - [x]` 任务列表 → `☐ / ☑` 转换，飞书旧版卡片不再丢这两种构造（之前只处理 `#` 标题、GFM 表格、有序列表）
- **空闲心跳** — 进程内 watchdog 每 5min 主动写一行 `xiaonz.alive`，刷 `agent.log` mtime；避免外部 launchd heartbeat 把"长时间没用户消息"的空闲进程误判为卡死并重启
- **`run_command` 拦截升级** — 现在也拦 `... & disown` 没有 stdout 重定向的模式（之前只拦裸 `&`），grandchild 继承 PIPE 后 `communicate()` 卡死的洞补齐；带 `>` / `>>` 显式重定向的安全形式（`nohup CMD >/tmp/svc.log 2>&1 </dev/null & disown`）仍然放行

## v0.2.0 更新一览

- **三层级联搜索** — `web_search` 从单层抓 DDG HTML 升级为 **Tavily → Brave → DuckDuckGo** 级联，任一层挂自动落下一层
  - Tavily AI 摘要（`include_answer`）置顶，简单问答能省一次 fetch
  - Brave 自动 CJK 检测 → 中文 query 优先返回中文站点
  - 新增 `freshness=day/week/month/year` 和 `topic=news` 入参，时效性查询大幅改善
- **重复调用熔断** — 模型连发同一工具相同参数 > 4 次自动判定死循环并停下，防止 token 被刷光
- **工具 180s 硬超时** — 外层 `asyncio.wait_for` 兜底，单工具死锁不再冻结整个 event loop
- **流式生成** — 模型调用改用 `messages.stream`，长输出不再被 httpx 切断报 ReadTimeout
- **Embedder 抗压** — 共享 httpx client + 信号量串行化 + 5xx 指数退避，单 worker bge-m3 不再被打爆；`trust_env=False` 自动绕开系统代理
- **错误友好化** — 飞书侧异常回复不再暴露 stack trace，给非技术用户看得懂的中文提示

---

## 功能一览

### Agent 能力（核心）

- **真正的 tool-use 主循环** — 多轮推理 + 调工具 + 看结果再继续，不是一问一答
- **23 个内置工具** — 上网、读 PDF/Word/Excel、跑 shell、读写文件、生图、无头浏览器抓正文 + 截图……
- **Skills 体系** — 兼容 [AgentSkills](https://agentskills.io) 标准，`install_skill` 按需装新能力
- **自主维护长期记忆** — Agent 自己用工具改写 `MEMORY.md`，每次写入自动备份（保留 30 份）
- **跨会话回忆** — 每天 cron 生成"昨日摘要" + 向量库 top-k 语义召回，能翻出几周前提过的事
- **死循环熔断** — 同工具同参数重复调用自动 warn → 强制 abort，模型刷不动 token
- **可中断** — 用户连发新消息会取消上一轮未完成的工具循环
- **流式状态可见** — 模型一边推理一边更新飞书卡片，不是干等

### 模型与上游

- 默认 **Claude Opus 4.7**，一行配置切到 Sonnet 4.6 / Haiku 4.5
- 任何 Anthropic Messages API 兼容上游（官方 / 自建网关 / 代理）
- **流式调用**避免长输出超时
- 自动会话压缩：长聊不爆 token

### 飞书集成

- **lark-oapi 长连接** — 无需公网入口，不用 VPS、不用 cloudflared
- 单聊 + 群聊（群里 @bot 触发）
- 卡片消息流式更新
- 收图片 / PDF / Word / Excel 等附件 — 自动转 Markdown 或按需调工具读
- Agent 可主动推消息 / 文件 / 图片到飞书
- 异常回复友好化，不暴露内部错误

### 工具一栏

| 类别 | 工具 |
|---|---|
| 记忆 | `update_memory` · `search_memory` · `search_memory_semantic` |
| 网络 | `web_fetch`（readability 正文提取）· `web_search`（**Tavily → Brave → DDG 级联**）· `download_to_disk` |
| 多模态 | `generate_image`（gpt-image-2 / DALL·E-3 兼容）· `read_pdf` · `anything_to_md` · `browser_capture`（无头浏览器截图 + 抓正文） |
| 飞书 | `send_to_feishu`（主动推图 / 文件 / 卡片） |
| 文件 | `read_local_file` · `write_local_file` · `create_directory` · `list_directory` · `move_path` · `copy_path` · `delete_path` |
| Shell | `run_command`（带超时 + 输出截断） |
| Skills | `list_skills` · `load_skill` · `install_skill` · `uninstall_skill` |

---

## 快速开始

### 1. 准备

- Python 3.9+
- 飞书自建应用（[飞书开放平台](https://open.feishu.cn/) 创建 → 启用"机器人"能力 → 拿到 `app_id` / `app_secret`）
- Anthropic API key（官方 [console.anthropic.com](https://console.anthropic.com/) 或任何兼容网关）
- *（可选）* [Tavily](https://tavily.com)（1000 次/月免费）和 [Brave Search](https://api.search.brave.com)（2000 次/月免费）API key —— 不配也能跑（自动降级到 DDG）

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

进程会前台运行 + 阻塞。生产环境建议用 `launchd` (macOS) / `systemd` (Linux) 守护。

---

## Web Search（三层级联）

`web_search` 工具按优先级依次尝试，前一层 ERROR 或空结果时自动落下一层：

| 层级 | 后端 | 免费配额 | 优势 |
|---|---|---|---|
| 1 | **Tavily** | 1000 次/月 | LLM-tuned 重排 + AI 摘要置顶，单条结果信息密度最高 |
| 2 | **Brave** | 2000 次/月（1 qps） | 独立索引日更，时效性强；中文 query 自动 zh-hans 定位 |
| 3 | **DuckDuckGo** | 无限（无 key） | 最终兜底，HTML 抓取 |

`config.yaml`：

```yaml
search:
  tavily_api_key: "tvly-..."   # 可选，留空跳过
  brave_api_key:  "BSA_..."    # 可选，留空跳过
```

调用入参支持 `freshness=day/week/month/year`（时间过滤）和 `topic=news`（Tavily news-tuned 索引），模型遇到突发事件 / 新发布项目时会主动用这两个参数收紧。

---

## Embedding（向量召回，可选）

默认**关闭**。开启后 Agent 每次回复前会做 top-k 语义召回（消息 + 每日 digest + MEMORY.md 分块），明显提升跨会话连贯性。

> v0.2.0 起 embedding 写入**不在请求路径上做**，统一由后台 cron 回填（见 `scripts/bootstrap_memory.py`）。请求路径只读不写，bge-m3 容器抖动不会拖慢回复。

### 方案 A：Ollama（推荐）

```bash
# macOS
brew install ollama
# 或 https://ollama.com/download

ollama pull bge-m3      # ~1.2GB
ollama serve            # 起服务在 :11434
```

`config.yaml`：

```yaml
embedding:
  enabled: true
  base_url: "http://127.0.0.1:11434/v1"
  model: "bge-m3"
  dim: 1024
```

### 方案 B：HuggingFace TEI 自部署

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) 模型，用 [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) 起服务后填 base_url 即可。

### 方案 C：OpenAI 官方 embeddings

```yaml
embedding:
  enabled: true
  base_url: "https://api.openai.com/v1"
  model: "text-embedding-3-large"
  dim: 3072
```

### 首次开启：bootstrap

启用后跑一次把已有 SOUL/MEMORY/历史消息全部 embed：

```bash
venv/bin/python scripts/bootstrap_memory.py
```

---

## 稳定性设计

v0.2.0 重点加固了几个容易翻车的点：

| 场景 | 兜底 |
|---|---|
| 模型连发同一工具陷入死循环 | 第 2 次注入 ⚠️ 提示，第 4 次直接 abort 当前 turn |
| 单个工具 await hang（playwright / httpx 都中过招） | `asyncio.wait_for(_, 180s)` 外层兜底 |
| 长生成被 httpx ReadTimeout 切断 | `messages.stream` + `get_final_message`，超时改为 per-chunk 阈值 |
| bge-m3 单 worker 被并发打 503 | 共享 client + Semaphore(1) 串行化 + 指数退避重试 |
| macOS 系统代理（ClashX）劫持 localhost | Embedder `trust_env=False` 强制绕开 |
| 异常直接拼 repr 抛回飞书 | `_friendly_error_text` 按类型映射成人话 |
| 飞书 post 富文本 / 语音 / 转发等被静默 skip | 全消息类型解析，照 [OpenClaw](https://github.com/openclaw/openclaw) 实现对齐 |
| heartbeat 把空闲进程误判卡死重启 | 进程内 watchdog 每 5min 主动写 `xiaonz.alive` 心跳日志 |
| `& disown` 不带 stdout 重定向卡 `communicate()` | `_has_bare_background_amp` 拦截 + 错误提示给安全形式 |

---

## 项目结构

```
src/
├── main.py              # 入口
├── config.py            # config.yaml loader
├── feishu/client.py     # 飞书长连接客户端
└── agent/
    ├── loop.py          # tool-use 主循环 + 重复熔断 + 硬超时
    ├── model_client.py  # Anthropic SDK 封装（流式）
    ├── memory.py        # MEMORY.md atomic 读写
    ├── vector_memory.py # SQLite + numpy 向量存储
    ├── embedder.py      # bge-m3 客户端（共享 client + 重试）
    ├── session.py       # SQLite 会话状态
    ├── skills.py        # AgentSkills 加载器
    └── tool_impls.py    # 工具实现（含三层级联搜索）

data/                    # 运行时数据（除模板外全 gitignore）
├── memory/
│   ├── SOUL.md.example
│   ├── MEMORY.md.example
│   └── README.md
└── (hermes_state.db, sessions/, generated/, ...)
```

---

## 模型切换

`config.yaml` 里改 `model.model_id`：

| 模型 | model_id |
|---|---|
| Claude Opus 4.7 | `claude-opus-4-7` |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` |

---

## 已知限制

- 单租户：一个进程一个用户。多用户场景请看 [OpenClaw](https://github.com/openclaw/openclaw)
- 仅飞书：其他 IM channel 没接。lark-oapi 替换成对应 SDK 即可，protocol 部分独立
- 工具受限：未走 MCP 标准，自定义工具需要在 `tool_impls.py` 注册

---

## License

[MIT](./LICENSE) © 2026 yiyizhi
