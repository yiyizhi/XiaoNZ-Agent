<div align="center">
  <img src=".github/readme-assets/logo.png" alt="小宁子" width="104" height="104" />
  <h1>小宁子 · XiaoNZ Agent</h1>
  <p><strong>住在你飞书里的 AI 助手，长在你自己的电脑上。</strong></p>
  <p>
    给它发条消息，它就能帮你查资料、读文件、画图、整理电脑里的东西——<br/>
    就像有个助理坐在你电脑前，而你只需要在飞书里说话。
  </p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB.svg" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/%E9%A3%9E%E4%B9%A6-%E9%95%BF%E8%BF%9E%E6%8E%A5%EF%BC%8C%E6%97%A0%E9%9C%80%E5%85%AC%E7%BD%91-00B96B.svg" alt="飞书长连接" />
  <img src="https://img.shields.io/badge/Claude-Messages%20API-d97757.svg" alt="Claude" />
  <img src="https://img.shields.io/badge/self--hosted-%E5%8D%95%E8%BF%9B%E7%A8%8B%EF%BC%8C%E9%9B%B6%E5%AE%B9%E5%99%A8-0ea5e9.svg" alt="Self-hosted" />
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT" />
</p>

<p align="center">
  <a href="#它能帮你做什么">能做什么</a> ·
  <a href="#搭建大约-10-分钟">快速搭建</a> ·
  <a href="#日常怎么用">日常怎么用</a> ·
  <a href="#让记性更好长期记忆可选推荐">长期记忆</a> ·
  <a href="#为什么它不容易死">可靠性设计</a>
</p>

---

<p align="center">
  <a href="https://github.com/yiyizhi/XiaoNZ-Agent/raw/main/.github/readme-assets/demo.mp4">
    <img src=".github/readme-assets/demo.gif" alt="小宁子对话演示" width="820" />
  </a>
</p>
<p align="center">
  <sub>▶ <b><a href="https://github.com/yiyizhi/XiaoNZ-Agent/raw/main/.github/readme-assets/demo.mp4">看 mp4 版演示</a></b>（演示动画，三段对话均为真实功能：联网搜索 · 读 PDF · 长期记忆召回）</sub>
</p>

不用买服务器，不用备案域名。一台不关机的 Mac 或 Linux 电脑，加一个免费的飞书自建应用，就能跑起来。

## 它能帮你做什么

直接看几段真实用法：

> **你**：明天要去杭州出差，帮我看下天气，要带伞吗
> **小宁子**：（自己上网搜完）明天杭州小雨转阴，18~24℃，建议带伞……

> **你**：（甩给它一份 30 页的 PDF 合同）帮我挑出对我不利的条款
> **小宁子**：（读完整份文件）有 3 处需要注意：第 4 条的违约金比例……

> **你**：把我桌面上的截图按月份整理到文件夹里
> **小宁子**：（在你电脑上动手）整理完了，共 47 张，分到了 2026-04 / 05 / 06 三个文件夹。

> **你**：画一张"程序员深夜改 bug"的漫画
> **小宁子**：（图片直接出现在聊天里）

> **你**：上上周我说想试的那家餐厅叫什么来着？
> **小宁子**：你 5 月 28 日提过，叫"山海小馆"，当时你说想约老张一起。

总结一下它的本事：

- **什么都能发给它**：文字、截图、PDF、Word、表格、语音、视频，它都看得懂
- **真的会动手**：上网搜索、打开网页、读写你电脑上的文件、执行命令、生成图片——23 个工具，它自己决定怎么组合着用
- **记性好**：重要的事它会记进小本本（一个它自己维护的记忆文件）；开启"长期记忆"后，几周前随口提过的事它也能想起来
- **能学新技能**：兼容 [AgentSkills](https://agentskills.io) 标准，聊天里发一句"安装这个技能"加个链接就行
- **皮实**：双层守护进程，崩了自动爬起来，假死了自动重启，不用你管

## ⚠️ 先说安全

它能在你电脑上跑命令、读写文件——**这意味着能给它发消息的人，约等于能操作你的电脑**。

所以它默认**谁都不理**：只有写进白名单（`allowed_open_ids`）的人发的消息才会被处理，没有"对所有人开放"这个选项。配置时这一步必做，下面会讲怎么配。

## 搭建（大约 10 分钟）

### 第 1 步：准备三样东西

1. **Python 3.11 或更新**（电脑上装好即可）
2. **一个飞书自建应用**（免费，5 分钟搞定）：
   - 到[飞书开放平台](https://open.feishu.cn/)创建应用，启用「机器人」能力
   - 事件订阅方式选「**长连接**」（这就是不需要服务器的原因——你的电脑主动连飞书，不用飞书来找你）
   - 订阅 `im.message.receive_v1` 事件
   - 记下 `app_id` 和 `app_secret`
3. **一个 Claude API 入口**：官方 [console.anthropic.com](https://console.anthropic.com/) 的 key，或任何兼容 Anthropic Messages API 的网关

> 搜索功能可以再配 [Tavily](https://tavily.com) / [Brave](https://api.search.brave.com) 的免费 key，效果更好；不配也能用（自动用 DuckDuckGo）。

### 第 2 步：下载安装

```bash
git clone https://github.com/yiyizhi/XiaoNZ-Agent.git
cd XiaoNZ-Agent

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 第 3 步：填配置

```bash
cp config.yaml.example           config.yaml
cp data/memory/SOUL.md.example   data/memory/SOUL.md
cp data/memory/MEMORY.md.example data/memory/MEMORY.md
```

打开 `config.yaml`，填三处：

```yaml
feishu:
  app_id: "cli_xxx"        # 第 1 步拿到的
  app_secret: "xxx"
  allowed_open_ids: []     # 白名单，先留空，下面教你怎么填

model:
  base_url: "https://api.anthropic.com"   # 或你的网关地址
  auth_token: "sk-xxx"
  model_id: "claude-fable-5"
```

**把自己加进白名单**：先启动一次（见第 4 步），给机器人发条消息——它不会理你，但 `agent.log` 里会出现一行 `feishu.sender_rejected`，里面的 `ou_xxx` 就是你的 ID。抄进 `allowed_open_ids`，重启即可。

另外 `SOUL.md` 是它的人设（说话风格、行为习惯），用文本编辑器打开，想怎么改就怎么改。

### 第 4 步：启动

```bash
python -m src.main
```

在飞书里给它发句"你好"，收到回复就是通了。🎉

想关掉就 Ctrl+C。想让它 7×24 小时跑着，看下一节。

## 让它一直在线（可选）

把它注册成系统服务，开机自启、崩了自动拉起。macOS 用 launchd（配置如下），Linux 用 systemd 同理。

<details>
<summary>点开看 macOS launchd 配置</summary>

**主服务** `~/Library/LaunchAgents/ai.xiaonz.agent.plist`（把 `/path/to` 换成你的实际路径）：

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
    <key>ThrottleInterval</key><integer>60</integer>
    <key>StandardOutPath</key><string>/path/to/XiaoNZ-Agent/agent.log</string>
    <key>StandardErrorPath</key><string>/path/to/XiaoNZ-Agent/agent.log</string>
</dict>
</plist>
```

**看门狗**（每 5 分钟检查一次，发现卡死就重启它，顺便清理日志）`~/Library/LaunchAgents/ai.xiaonz.heartbeat.plist`，关键两行：

```xml
<key>ProgramArguments</key>
<array><string>/path/to/XiaoNZ-Agent/scripts/heartbeat.sh</string></array>
<key>StartInterval</key><integer>300</integer>
```

注册生效：

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.xiaonz.agent.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.xiaonz.heartbeat.plist
```

</details>

## 日常怎么用

私聊直接说话，群里 @ 它。几个聊天命令：

| 发什么 | 效果 |
|---|---|
| `/new` | 重新开始一段对话（忘掉当前上下文） |
| `/cancel` | 停下正在做的事（撤回你刚发的消息也有同样效果） |
| `/mem` | 看看它的小本本上记了你哪些事 |
| `/skills` | 看已安装的技能 |
| `/help` | 帮助 |

## 让记性更好：长期记忆（可选，推荐）

默认情况下它只记得"小本本"上的内容。打开长期记忆后，它每天会自动把聊天内容做成可检索的索引——"上上周说的那家餐厅"这种问题就能答上来。

需要一个文本向量化服务，三选一：

<details>
<summary>点开看三种方案</summary>

**方案 A：TEI + Docker（推荐，速度快）**

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

**方案 C：直接用 OpenAI 官方接口**

`config.yaml` 里 `base_url` 填 `https://api.openai.com/v1`，`model` 填 `text-embedding-3-large`，`dim` 填 `3072`。

</details>

然后在 `config.yaml` 打开开关，并把历史聊天记录建一次索引：

```yaml
embedding:
  enabled: true
  base_url: "http://127.0.0.1:11434/v1"
  model: "bge-m3"
  dim: 1024
```

```bash
venv/bin/python scripts/bootstrap_memory.py   # 只需跑这一次，之后每天自动更新
```

放心开：这个服务就算挂了，也只影响"翻旧账"，正常聊天不受任何影响。

## 为什么它不容易死

个人助手最怕的不是不聪明，是"叫不应"。这套是长期真实运行踩坑后攒下的保命机制，全部默认开启：

| 它防的事 | 怎么防 |
|---|---|
| 飞书连接抖动、假死 | 进程内自检 + 外部看门狗双保险，连接异常自动重连，叫不醒就整个重启 |
| AI 接口卡住不返回 | 90 秒没动静就自动换线路重试，整轮最多等 10 分钟，绝不无限挂着 |
| 某个任务把它拖死 | 每个工具最多跑 180 秒；干重活不影响它同时接收新消息 |
| 模型陷入死循环 | 同一个工具用同样参数连续调 4 次以上，自动熔断 |
| 电脑上的代理软件挂了 | 飞书和 AI 接口都强制直连，不经过系统代理 |
| 长期运行越跑越慢 | 每 8 小时挑空闲时间自我重启一次 |

## 全部工具（给好奇的人）

<details>
<summary>点开看 23 个内置工具</summary>

| 类别 | 工具 |
|---|---|
| 记忆 | `update_memory` · `search_memory` · `search_memory_semantic` |
| 网络 | `web_search`（Tavily → Brave → DuckDuckGo 自动降级）· `web_fetch` · `download_to_disk` |
| 多模态 | `generate_image` · `read_pdf` · `anything_to_md`（docx/xlsx/pptx/epub 转 markdown）· `browser_capture`（无头浏览器截图+抓正文） |
| 飞书 | `send_to_feishu`（主动推图 / 文件 / 卡片） |
| 文件 | `read_local_file` · `write_local_file` · `create_directory` · `list_directory` · `move_path` · `copy_path` · `delete_path`（带系统关键路径保护） |
| Shell | `run_command`（120 秒超时 + 进程组清理 + 输出截断） |
| 技能 | `list_skills` · `load_skill` · `install_skill` · `uninstall_skill` |

</details>

## 代码结构（给想改代码的人）

<details>
<summary>点开看目录说明</summary>

```text
src/
  main.py            入口：装配 + 信号处理 + 代理豁免
  config.py          配置加载（pydantic 严格校验，拼错字段启动即报错）
  feishu/client.py   飞书长连接、事件分发、watchdog、卡片收发
  agent/
    loop.py          Agent 主循环（tool-use 多轮推理、压缩、熔断）
    model_client.py  Anthropic 流式客户端（超时自救 + 会话轮换）
    session.py       SQLite 会话存储（WAL，崩溃安全）
    tool_impls.py    全部内置工具实现
    memory.py        MEMORY.md 读写（原子写 + 自动备份）
    vector_memory.py embedding 向量库与召回
    skills.py        AgentSkills 加载
    commands.py      /new /cancel 等聊天命令
scripts/
  heartbeat.sh         外部看门狗：假死重载 + 日志轮转 + 产物清理
  bootstrap_memory.py  长期记忆索引的全量/增量回填
data/                  运行时数据（不入库）：SQLite、记忆、技能、附件
```

</details>

## License

[MIT](./LICENSE) © 2026 yiyizhi
