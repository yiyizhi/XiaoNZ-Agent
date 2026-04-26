# XiaoNZ Agent

> 轻量化的 [OpenClaw](https://github.com/openclaw/openclaw) —— 一个跑在飞书里的个人 AI 助手，原生接 **Claude Opus 4.7**。
> Single Python process · 飞书长连接 · SQLite 持久化 · 可选向量记忆。

**v0.1.0** · [MIT License](./LICENSE)

---

## 这是什么

XiaoNZ Agent 是一个长期运行的个人 Agent。它：

- 通过**飞书长连接**收发消息（无需公网 HTTPS 入口）
- 把对话、长期记忆、人设全存在本地 **SQLite + Markdown 文件**里
- 调用 **Claude Opus 4.7**（或任何 Anthropic Messages API 兼容端点）
- 每天自动生成"昨日 digest"并写入向量库，让 Agent 能跨天回忆

跟完整版 OpenClaw 的区别：**不分布式、不多租户、不容器化**——一台机器一个用户一个进程，简单粗暴，核心代码 ~3k 行能看完。

---

## 已实现功能

### 🧠 模型 & 对话
- **Claude Opus 4.7 默认**，可在 config 里切到 Sonnet 4.6 / Haiku 4.5
- 任何 Anthropic Messages API 兼容上游（官方 / 自建网关 / 代理）
- Tool-use 主循环（最多 20 轮 / 每轮用户消息）
- 自动会话压缩：history 超过阈值时把旧消息折叠成 summary
- 取消机制：用户连发新消息会取消上一轮未完成的工具循环

### 💬 飞书集成
- lark-oapi 长连接客户端，无需公网入口
- 单聊 + 群聊（@bot 触发）
- 卡片消息：模型一边生成一边 patch 卡片（流式体验）
- 接收消息 + 图片 + 文件附件（自动转 Markdown 注入上下文）
- "我想想…" 思考中状态指示
- 大附件自动落盘 + 通过 `read_pdf` / `anything_to_md` 工具按需读

### 📚 记忆系统
- **SOUL.md** — 人设（你创建，长期不变，每轮注入 system prompt）
- **MEMORY.md** — 长期记忆（Agent 通过 `update_memory` 工具自主维护，每次写入前自动 archive 备份，保留最近 30 份）
- **每日 digest** — 每天首次对话时自动生成"昨日摘要"并 embed 入向量库
- **会话历史** — SQLite 持久化，断电重启不丢
- **向量召回（可选）** — bge-m3 1024-d 向量，覆盖消息 / digest / Markdown 分块；每轮回复前 top-k 语义召回注入

### 🛠️ 工具集（23 个）

| 类别 | 工具 |
|---|---|
| **记忆** | `update_memory` · `search_memory` · `search_memory_semantic` |
| **网络** | `web_fetch`（含 readability 正文提取）· `web_search` · `download_to_disk` |
| **多模态** | `generate_image`（gpt-image-2 / DALL·E-3 兼容）· `read_pdf` · `anything_to_md` · `browser_capture`（无头浏览器截图 + 抓正文） |
| **飞书** | `send_to_feishu`（主动推图 / 文件 / 卡片） |
| **文件** | `read_local_file` · `write_local_file` · `create_directory` · `list_directory` · `move_path` · `copy_path` · `delete_path` |
| **Shell** | `run_command`（带超时 + 输出截断） |
| **Skills** | `list_skills` · `load_skill` · `install_skill` · `uninstall_skill`（[AgentSkills](https://agentskills.io) 标准格式） |

### ⚙️ 工程细节
- httpx 细分 timeout（connect 10s / read 60s）+ 单次重试，避免上游慢导致事件循环假死
- SQLite 写入走 `asyncio.to_thread`，不阻塞主循环
- Fire-and-forget 后台任务统一异常捕获 + 日志，不会静默死掉
- `LaunchAgent` (macOS) / `systemd` (Linux) 友好的前台阻塞模式

---

## 快速开始

### 1. 准备

- Python 3.9+
- 飞书自建应用（[飞书开放平台](https://open.feishu.cn/) 创建 → 启用"机器人"能力 → 拿到 `app_id` / `app_secret`）
- Anthropic API key（官方 [console.anthropic.com](https://console.anthropic.com/) 或任何兼容网关）

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

打开 `config.yaml` 填飞书凭证 + Anthropic key。`SOUL.md` 按你的口味改写人设。

### 4. 启动

```bash
python -m src.main
```

进程会前台运行 + 阻塞。生产环境建议用 `launchd` (macOS) / `systemd` (Linux) 守护。

---

## Embedding（向量召回，可选）

默认**关闭**。开启后 Agent 每次回复前会做 top-k 语义召回（消息 + 每日 digest + MEMORY.md 分块），明显提升跨会话连贯性。

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

## 项目结构

```
src/
├── main.py              # 入口
├── config.py            # config.yaml loader
├── feishu/client.py     # 飞书长连接客户端
└── agent/
    ├── loop.py          # tool-use 主循环
    ├── model_client.py  # Anthropic SDK 封装
    ├── memory.py        # MEMORY.md atomic 读写
    ├── vector_memory.py # SQLite + numpy 向量存储
    ├── embedder.py      # bge-m3 客户端
    ├── session.py       # SQLite 会话状态
    ├── skills.py        # AgentSkills 加载器
    └── tool_impls.py    # 工具实现

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
