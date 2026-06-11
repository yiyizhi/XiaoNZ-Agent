# XiaoNZ Agent

一个跑在飞书里的个人 AI 助手。在飞书里给它发消息，它能上网查资料、读你发的文件、在你的电脑上干活、生成图片、记住你的长期偏好，还能跨越几周回忆起以前聊过的事。

不需要服务器、不需要公网域名——一台常开的 Mac / Linux 机器 + 一个飞书自建应用就能跑。

```text
你在飞书发消息（文字 / 图片 / 文件 / 语音 / 视频）
        ↓ 飞书长连接（无需公网入口）
Agent loop（Claude，多轮思考 + 调工具）
        ↓
搜索 · 抓网页 · 读文档 · 跑命令 · 生图 · 记忆召回 …
        ↓
回复卡片 / 文件 / 图片 推回飞书
```

## 能干什么

- **聊天即用**：私聊直接说话，群里 @ 它。发什么都行——文字、截图、PDF、Word、表格、语音、视频。
- **真的会干活**：23 个内置工具，模型自己决定调用顺序——搜索（Tavily → Brave → DuckDuckGo 三层自动降级）、抓网页正文、无头浏览器截图、读写本地文件、跑 shell 命令、生成图片并直接发到聊天里。
- **记得住你**：
  - 长期记忆是一个它自己维护的 `MEMORY.md`（每次改写自动备份，保留 30 份）；
  - 开启向量召回后，每次回复前先做语义检索（历史消息 + 每日摘要 + 记忆分块），几周前提过的事也翻得出来。
- **能学新技能**：兼容 [AgentSkills](https://agentskills.io) 标准（SKILL.md），聊天里发一句"安装这个技能 + 链接"就能装。
- **不容易死**：进程内 watchdog + 外部 heartbeat 双层守护，崩了自动拉起、假死自动重载（见[可靠性设计](#可靠性设计)）。

## 聊天命令

| 命令 | 作用 |
|---|---|
| `/new` | 开始新对话（先中断进行中的请求，再清空当前会话历史） |
| `/cancel` | 中断正在处理的请求（也可以直接撤回你发的那条消息） |
| `/mem` | 查看长期记忆 MEMORY.md |
| `/skills` | 列出已安装技能 |
| `/help` | 帮助 |

## 快速开始

### 0. 准备

- Python **3.11+**（开发机实际跑 3.13）
- 飞书自建应用：[飞书开放平台](https://open.feishu.cn/) 创建应用 → 启用「机器人」能力 → 事件订阅选**长连接模式** → 订阅 `im.message.receive_v1` → 拿到 `app_id` / `app_secret`
- 一个 Anthropic Messages API 端点：官方 [console.anthropic.com](https://console.anthropic.com/) 的 key，或任何兼容网关
- *（可选）* [Tavily](https://tavily.com) / [Brave Search](https://api.search.brave.com) 搜索 key——不配也能跑，自动降级到 DuckDuckGo

### 1. 安装

```bash
git clone https://github.com/yiyizhi/XiaoNZ-Agent.git
cd XiaoNZ-Agent

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt     # 或 requirements.lock 复现完全一致的版本
```

### 2. 配置

```bash
cp config.yaml.example           config.yaml
cp data/memory/SOUL.md.example   data/memory/SOUL.md
cp data/memory/MEMORY.md.example data/memory/MEMORY.md
```

打开 `config.yaml`，三件事：

1. 填 `feishu.app_id` / `app_secret`；
2. 填 `model.base_url` / `auth_token` / `model_id`；
3. **配置 `allowed_open_ids` 白名单**（必须）。

> ⚠️ **安全模型**：这个 bot 能在你机器上跑 shell、读写文件。所以白名单是**默认拒绝**的——名单为空时谁的消息都不处理，没有"对所有人开放"模式。第一次拿不到自己的 open_id？先启动，给 bot 发条消息，去 `agent.log` 里找 `feishu.sender_rejected` 那行，把里面的 `ou_xxx` 抄进白名单重启即可。

`SOUL.md` 是人设和行为守则，按口味改写。

### 3. 启动

```bash
python -m src.main
```

前台运行。给 bot 发条消息试试，看到回复就通了。长期运行请配守护进程（下一节）。

## 部署：守护进程

以 macOS launchd 为例（Linux 用 systemd 同理，要点一样）：

**主服务** `~/Library/LaunchAgents/ai.xiaonz.agent.plist`：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTD/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>ai.xiaonz.agent</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/XiaoNZ-Agent/venv/bin/python</string>
        <string>-m</string>
        <string>src.main</string>
    </array>
    <key>WorkingDirectory</key><string>/path/to/XiaoNZ-Agent</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <!-- 启动即崩（如配置写错）时的重拉间隔，别设太短防崩溃循环刷日志 -->
    <key>ThrottleInterval</key><integer>60</integer>
    <key>StandardOutPath</key><string>/path/to/XiaoNZ-Agent/agent.log</string>
    <key>StandardErrorPath</key><string>/path/to/XiaoNZ-Agent/agent.log</string>
</dict>
</plist>
```

**外部心跳**（每 5 分钟跑一次 `scripts/heartbeat.sh`，负责假死重载 + 日志轮转 + 旧产物清理）：

```xml
<key>ProgramArguments</key>
<array><string>/path/to/XiaoNZ-Agent/scripts/heartbeat.sh</string></array>
<key>StartInterval</key><integer>300</integer>
```

加载：

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.xiaonz.agent.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.xiaonz.heartbeat.plist
```

## 可靠性设计

实际长期运行踩坑后沉淀的多层防护，全部默认开启：

| 层 | 机制 | 针对什么故障 |
|---|---|---|
| 进程内 watchdog | ws 反复重连（90s 内 ≥3 次）→ 自杀由 launchd 拉起 | 飞书长连接抖动 |
| 进程内 watchdog | 每 8h 预防性重启（有对话在处理就推迟，最多宽限 30 分钟） | 慢性资源泄漏 |
| 进程内 watchdog | 连续 4h 零事件 → 停写 alive 日志，移交外部心跳 | **软僵尸**（ws 自称在线但收不到消息） |
| 外部 heartbeat | 日志 10 分钟没动静 → `bootout + bootstrap` 完整重载 | 进程卡死 / 软僵尸（实测只有完整重载救得回） |
| LLM 调用 | 流式 + 逐 chunk 超时 + **整轮 600s 墙钟硬上限** + 连接级重试 | 上游网关卡顿 / 滴流 / 瞬断 |
| 工具执行 | 单工具 180s 硬超时；重活全部在线程池跑，不堵事件循环 | 大文件 / 慢网页拖死整个 bot |
| 会话 | 按 session 串行 + 事件去重 + 死循环熔断（同工具同参数 >4 次 abort） | 并发写坏历史 / 模型刷 token |
| 代理环境 | 飞书域名与网关 IP 强制直连（`no_proxy`），不受系统代理软件影响 | Clash 等代理挂掉连带 bot 失联 |

## 向量召回（可选，推荐）

默认关闭。开启后明显提升跨会话连贯性："上上周说的那家餐厅"这种话它能接住。

embedding 写入由每日定时任务回填，**请求路径只读不写**——embedding 服务挂了不影响正常聊天。

任选一个 OpenAI 兼容的 `/v1/embeddings` 服务：

**方案 A：TEI（推荐，批量快 5~10x）**

```bash
docker run -d --name tei-bge-m3 \
  -p 11434:80 \
  -v ~/.cache/huggingface:/data \
  ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
  --model-id BAAI/bge-m3
```

**方案 B：Ollama（不想用 Docker）**

```bash
brew install ollama
ollama pull bge-m3
ollama serve
```

**方案 C：OpenAI 官方**（`base_url: https://api.openai.com/v1`，`model: text-embedding-3-large`，`dim: 3072`）

`config.yaml` 打开开关：

```yaml
embedding:
  enabled: true
  base_url: "http://127.0.0.1:11434/v1"
  model: "bge-m3"
  dim: 1024
```

首次开启跑一遍全量回填（之后每天自动增量）：

```bash
venv/bin/python scripts/bootstrap_memory.py
```

## 工具一览

| 类别 | 工具 |
|---|---|
| 记忆 | `update_memory` · `search_memory` · `search_memory_semantic` |
| 网络 | `web_search`（三层级联，支持 freshness/news）· `web_fetch`（正文提取）· `download_to_disk` |
| 多模态 | `generate_image` · `read_pdf` · `anything_to_md`（docx/xlsx/pptx/epub→markdown）· `browser_capture`（无头浏览器截图+抓正文） |
| 飞书 | `send_to_feishu`（主动推图 / 文件 / 卡片） |
| 文件 | `read_local_file` · `write_local_file` · `create_directory` · `list_directory` · `move_path` · `copy_path` · `delete_path`（带系统关键路径保护） |
| Shell | `run_command`（120s 超时 + 进程组清理 + 输出截断） |
| Skills | `list_skills` · `load_skill` · `install_skill` · `uninstall_skill` |

## 项目结构

```text
src/
  main.py            入口：装配 + 信号处理 + 代理豁免
  config.py          配置加载（pydantic 严格校验，拼错字段启动即报错）
  feishu/client.py   飞书长连接、事件分发、watchdog、卡片收发
  agent/
    loop.py          Agent 主循环（tool-use 多轮推理、压缩、熔断）
    model_client.py  Anthropic 流式客户端（整轮超时兜底）
    session.py       SQLite 会话存储（WAL，崩溃安全）
    tool_impls.py    全部内置工具实现
    memory.py        MEMORY.md 读写（原子写 + 自动备份）
    vector_memory.py embedding 向量库与召回
    skills.py        AgentSkills 加载
    commands.py      /new /cancel 等聊天命令
scripts/
  heartbeat.sh         外部心跳：假死重载 + 日志轮转 + 产物清理
  bootstrap_memory.py  向量库全量/增量回填（配合每日定时任务）
data/                  运行时数据（gitignored）：SQLite、记忆、技能、附件
```

## License

[MIT](./LICENSE) © 2026 yiyizhi
