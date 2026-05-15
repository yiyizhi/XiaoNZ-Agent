#!/bin/bash
# Watchdog for the xiaonz agent. Runs every 5 minutes via launchd.
#
# 历史教训（2026-05-12）：
#   `launchctl kickstart -k` 重启进程，但飞书消息回调进不到 agent loop——
#   ws 层 lark sdk 仍打 "Lark connected"，feishu.recv 全部不触发。
#   实测 21h48m 期间被 kickstart -k 踢 18 次都没救回，必须改成完整
#   bootout + bootstrap（重载 launchd job，重建 lark sdk）才能恢复消息流。
#
# 改动：
# - kickstart -k → bootout + bootstrap（彻底重新加载 plist）
# - SILENCE_THRESHOLD 30min → 10min（卡死 → 自愈时长上限）

set -u

# 解析项目根目录，无需硬编码用户路径（脚本在 PROJECT_ROOT/scripts/ 下）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG="$PROJECT_ROOT/agent.log"
HEARTBEAT_LOG="$PROJECT_ROOT/heartbeat.log"
LABEL="ai.xiaonz.agent"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
SILENCE_THRESHOLD=600   # 10 minutes (was 1800s; kickstart 救不回，要快速 reload)

if [ ! -f "$LOG" ]; then
    exit 0
fi

NOW=$(date +%s)
MTIME=$(stat -f %m "$LOG" 2>/dev/null)
if [ -z "$MTIME" ]; then
    exit 0
fi

DIFF=$((NOW - MTIME))
if [ "$DIFF" -gt "$SILENCE_THRESHOLD" ]; then
    PID=$(launchctl list | awk -v lbl="$LABEL" '$3==lbl {print $1}')
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] silent=${DIFF}s pid=${PID:-none} -> bootout+bootstrap" >> "$HEARTBEAT_LOG"
    # bootout 不阻塞；如果进程没在 5s 内退出，强 kill 残留再 bootstrap
    launchctl bootout "gui/$(id -u)/${LABEL}" 2>>"$HEARTBEAT_LOG"
    for i in 1 2 3 4 5; do
        sleep 1
        STILL=$(launchctl list | awk -v lbl="$LABEL" '$3==lbl {print $1}')
        [ -z "$STILL" ] && break
    done
    # 最后兜底：旧 PID 还在就 SIGKILL（memory 已记 SIGTERM 干不掉的情况）
    if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] stale pid=$PID still alive -> SIGKILL" >> "$HEARTBEAT_LOG"
        kill -9 "$PID" 2>>"$HEARTBEAT_LOG"
    fi
    launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>>"$HEARTBEAT_LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] bootstrap done" >> "$HEARTBEAT_LOG"
fi
