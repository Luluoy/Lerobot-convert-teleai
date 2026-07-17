# 安装引导

最后验证日期：2026-07-17。

本引导用于安装 LeRobot Data Convert 后端与 PWA。若安装方式、依赖、启动命令或 systemd
配置发生变化，必须在同一次改动中更新本文件和
`skills/maintain-lerobot-converter/references/operations-and-ui.md`，避免后续 Agent 按过时步骤操作。

## 1. 环境要求

- Linux，Python 3.10 或更高版本。
- Git。
- 能够编码 AV1 的 FFmpeg；推荐包含 `libsvtav1` 编码器。
- user systemd 会话，用于登录后自动启动后端（可选但推荐）。
- Node.js 仅用于运行前端端到端测试，正常使用不需要 Node.js。

可先检查：

```bash
python3 --version
ffmpeg -version
ffmpeg -encoders | grep -E 'AV1|av1'
systemctl --user show-environment
```

## 2. 获取代码

```bash
git clone https://github.com/Luluoy/Lerobot-convert-teleai.git
cd Lerobot-convert-teleai
```

## 3. 安装 Python 依赖

推荐使用项目内虚拟环境，`start.sh` 会自动优先使用它：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

项目固定使用 LeRobot `0.3.3`。不要在未验证转换、合并和预览兼容性的情况下单独升级
LeRobot。

如需运行浏览器测试：

```bash
npm ci
```

`tests/ui_check.mjs` 当前使用 `/usr/bin/google-chrome`。在其他环境运行测试时，需要安装该
浏览器或同步调整测试启动路径。

## 4. 手动启动与验证

```bash
./start.sh
```

默认地址为 `http://127.0.0.1:8765`。保持后端终端运行，然后验证：

```bash
curl -fsS http://127.0.0.1:8765/api/health
```

如需指定解释器、端口或状态目录：

```bash
LEROBOT_DATACONVERT_PYTHON=/absolute/path/to/python \
LEROBOT_DATACONVERT_STATE=/absolute/path/to/state \
./start.sh --host 127.0.0.1 --port 8765
```

不要在没有访问控制的情况下把服务绑定到公网地址。

## 5. 安装 user systemd 服务

推荐配置 user service，使安装后的 PWA 打开时后端已经运行：

```bash
./install-systemd-service.sh
```

安装脚本会生成并启用：

```text
~/.config/systemd/user/lerobot-dataconvert.service
```

常用命令：

```bash
systemctl --user status lerobot-dataconvert
systemctl --user restart lerobot-dataconvert
systemctl --user stop lerobot-dataconvert
journalctl --user -u lerobot-dataconvert -f
```

若 PWA 显示“后端未连接”：

1. 首次部署时，让 Agent 按本文件完成环境和服务安装。
2. 已安装时，运行 `systemctl --user restart lerobot-dataconvert`。
3. 检查 `systemctl --user status` 和 `journalctl` 输出，再在页面点击“重新检测”。

PWA 本身不能执行系统命令，也不能在未运行后端时自行启动后端。

## 6. 安装 PWA

后端运行时，在 Chromium/Chrome 中打开 `http://127.0.0.1:8765`，使用浏览器安装入口安装
PWA。PWA shell 可以离线打开，但扫描、预览、转换和任务管理都需要后端在线。

## 7. 数据与安全

- 默认任务数据库位于 `~/.local/share/lerobot-dataconvert/jobs.sqlite3`。
- 转换恢复 cache 位于输出目录旁的 `.<输出目录名>.lerobot-cache`。
- “从列表删除任务”只删除任务数据库记录，不修改原始数据、输出数据或恢复 cache。
- CPU 核心数和占用上限分别控制并发范围与 duty cycle；占用上限最高为 95%。
- TeleAxis 数据包含 Python pickle，只能加载可信的本地采集数据。

## 8. 验证安装

```bash
python -m compileall -q lerobot_dataconvert tests
python -m unittest -v
```

安装了 Node.js 和 Chrome 后可继续运行：

```bash
node tests/ui_check.mjs
```

## 9. 更新

PWA 会在后端连接后自动检查远端。仅当工作区干净、本地没有独有提交且远端有新提交时，
才会显示“拉取更新”；该按钮只执行 fast-forward pull。检测到本地修改时，自动更新会
持久暂停，请寻求技术帮助或询问 Agent。处理完修改后，必须在页面手动点击“检查远端
更新”才能恢复检查。

页面拉取完成后，或已经手动完成 `git pull --ff-only` 后，在项目根目录执行：

```bash
./apply-update.sh
```

脚本本身不会执行 Git 操作。它会拒绝脏工作区和活动转换任务，使用与 `start.sh` 相同的
Python 环境安装声明的依赖，重启现有 systemd 用户服务，并等待 `/api/health` 恢复。

完整的手动更新顺序是：

```bash
git pull --ff-only
./apply-update.sh
```

正常运行不依赖 Node.js；只有需要运行前端测试时才执行 `npm ci`。
