# `data/memory/`

Agent 的长期记忆目录。

## 文件

| 文件 | 用途 |
|---|---|
| `SOUL.md` | Agent 人设（你创建，长期不变） |
| `MEMORY.md` | 长期记忆（Agent 自主维护，可手动编辑） |
| `archive/` | `MEMORY.md` 每次写入前的时间戳快照（保留最近 30 份） |

## 上手

```bash
cp data/memory/SOUL.md.example   data/memory/SOUL.md
cp data/memory/MEMORY.md.example data/memory/MEMORY.md
```

按需修改两个文件，然后启动 Agent。后续 `MEMORY.md` 由 Agent 通过 `update_memory` 工具自己更新；你想手动改也可以，Agent 会读到。

## 向量召回（可选）

启用 `config.yaml` 里的 `embedding` 后，Agent 会把每条消息、每日 digest、以及 SOUL/MEMORY 的分块写入 SQLite 向量表，每次回复前做 top-k 语义召回。详见 [README.md](../../README.md#embedding-向量召回可选)。

首次启用时跑一次 bootstrap 把已有记忆导入：

```bash
venv/bin/python scripts/bootstrap_memory.py
```
