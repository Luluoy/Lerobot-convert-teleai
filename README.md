# LeRobot Data Convert

本地多进程 LeRobot 数据转换工作台。后端使用 Python、SQLite 和原子 JSON manifest，前端是无构建步骤的 PWA。

首次部署请阅读 [`INSTALL.md`](./INSTALL.md)。后续 Agent 修改本工程前必须遵循
[`skills/maintain-lerobot-converter/SKILL.md`](./skills/maintain-lerobot-converter/SKILL.md)。

## 启动

本机已有的 `lerobot21` Conda 环境包含运行依赖：

```bash
cd ~/.coding-ground/loy/lerobot_dataconvert
./start.sh
```

打开 `http://127.0.0.1:8765`。如需使用其他 Python：

```bash
LEROBOT_DATACONVERT_PYTHON=/path/to/python ./start.sh --port 8765
```

安装为随当前用户登录自动启动的 systemd 服务：

```bash
./install-systemd-service.sh
```

服务日志和常用管理命令：

```bash
systemctl --user status lerobot-dataconvert
journalctl --user -u lerobot-dataconvert -f
systemctl --user restart lerobot-dataconvert
systemctl --user stop lerobot-dataconvert
```

## 代码更新

后端从 Git checkout 启动时，PWA 会异步检查当前分支的远端跟踪分支。工作区干净且
本地仅落后远端时，页面提供“拉取更新”，后端只执行 `git pull --ff-only`。页面拉取或
手动执行 `git pull --ff-only` 后，在项目根目录运行：

```bash
./apply-update.sh
```

该脚本不会操作 Git；它会确认工作区干净且没有活动转换任务，安装声明的 Python 依赖，
重启已安装的 systemd 用户服务，并等待后端恢复健康。运行前端测试时再单独执行 `npm ci`。

若检测到任何已跟踪或未跟踪的本地修改，平台会把暂停状态记录到运行状态目录的
`git-update-state.json`，停止自动检查与更新，并提示寻求技术帮助或询问 Agent。即使
本地修改随后消失，自动检查也不会自行恢复；必须手动点击“检查远端更新”。平台不会
自动 stash、reset、覆盖本地修改或创建 merge commit。

也可以安装到一个含 LeRobot 0.3.3 的环境：

```bash
python -m pip install -e .
lerobot-dataconvert
```

## 转换模型

每个任务被切分为若干小 segment。每个 worker：

1. 只读取分配给自己的原始 episodes。
2. 写入独立的 LeRobot v2.1 segment 目录。
3. 完成所有视频、Parquet 和元数据后才原子写入 `.segment-complete.json`。

主进程只认可带完成标记的 segment。正常停止会终止当前 worker 并删除未完成 segment；进程或机器异常退出后，服务启动时执行同一套清理并自动重新排队。已完成 segment 不会重复转换。

Cache 位于输出目录旁边：

```text
/datasets/task-a                  # 最终输出
/datasets/.task-a.lerobot-cache  # 恢复 manifest 与临时 segment
```

选择 v3.0 时，原始数据仍只转换一次。完成的 v2.1 segments 先合并，再在可中断的 finalizer 进程中打包为 v3.0；半成品 finalizer 目录在恢复时直接丢弃。

CPU 核心滑杆决定 worker 数以及可用 CPU 集，每个 worker 固定到一个核心；CPU 占用上限可向下调节但最高为 95%，调度器会同时限速 worker 及其编码子进程。内存预算结合适配器给出的单 worker 估算限制并发数，任务详情显示实际 worker 数和实时 RSS。

## 原始数据适配器

任何格式只需实现 [`RawDatasetAdapter`](./lerobot_dataconvert/adapters.py) 的三个方法。`inspect()` 返回的 `DatasetDescriptor.fields` 声明原始字段名称、shape、dtype、是否为 state/action/image、分量名称和默认 LeRobot 目标字段；这些是每个字段的独立属性，同类字段可有多个。`iter_frames()` 在 `FrameSample.fields` 中提供对应值：

```python
from lerobot_dataconvert.adapters import RawDatasetAdapter, register_adapter
from lerobot_dataconvert.models import RawField


@register_adapter
class MyAdapter(RawDatasetAdapter):
    slug = "my_format"
    display_name = "My Format"
    description = "My robot recording layout."

    def inspect(self):
        # DatasetDescriptor(..., fields=[
        #     RawField("robot/qpos", (14,), default_target="observation.state",
        #              is_state=True),
        #     RawField("robot/qvel", (14,), is_state=True),
        #     RawField("robot/action", (14,), default_target="action", is_action=True),
        #     RawField("front", (480, 640, 3), "uint8", True,
        #              "observation.images.front"),
        # ])
        ...

    def iter_frames(self, episode):
        # Yield FrameSample(..., fields={"robot/qpos": qpos, "front": rgb}).
        ...

    def preview(self, episode, camera, frame_index):
        # Return one RGB uint8 HWC image.
        ...
```

动作过滤使用所有 `is_action=True` 的原始字段；任意一个字段变化即视为发生动作。可选的
全 0 填充会检查所有 `is_state=True` 或 `is_action=True` 的字段：某字段整组元素严格等于
0 时，使用同一 episode 上一原始帧该字段的有效值；episode 首帧保持原值。该处理先于
动作过滤，因此预扫描与转换结果一致。基类的
`iter_action_values()` 默认复用 `iter_frames()`，适配器可覆写该方法，只读取 action 流，
避免预扫描时解码图像。每个 episode 始终至少保留一帧。

扫描后，PWA 的字段映射列表保持为空。使用加号逐行添加映射：每行从适配器声明的
原始字段（包含类型、shape 和 FPS）中选择来源，再选择或输入 LeRobot 目标 feature。
同一原始字段可添加多次并写入不同目标，LeRobot 目标字段必须唯一；不需要的字段不添加。
映射按行顺序写入 cache，恢复时保持不变。Cache、进程调度、资源限制、LeRobot 写入、
revision、合并和 ETA 不需要在适配器中实现。

把适配器模块加入环境变量即可在服务启动时加载：

```bash
LEROBOT_DATACONVERT_PLUGINS=my_project.my_adapter ./start.sh
```

可安装的 Python 包也可以声明 `lerobot_dataconvert.adapters` entry point，entry point 值指向适配器类。

内置 `hdf5_joint` 适配器支持：

- 一个 HDF5 文件对应一个 episode。
- 一个目录对应一个 episode，目录内每个 HDF5 文件对应一帧。
- `observations/qpos` / `action` schema。
- `puppet/joint_position` / `master/joint_position` schema。
- JPEG buffer 或原始 HWC RGB 图像。

内置 `multiprocessing_pool_dataset`（界面名称 `MultiProcessing Pool Dataset`）适配器支持 TeleAxis Collector schema v3：

- `episode_XXXXXX/META/meta.json` 中状态为 `complete` 且没有保存错误的 episode。
- `joint_state` 的 `qpos/qvel/torque`、`joint_action/action` 和 `eef_action/action` 逐帧 PKL 字段。
- `Cam*` PNG 相机流，以及 META 中配置的任意相机数据集名称。
- 每个原始字段都声明自己的 FPS；优先读取 META 的正数 `actual_fps`，否则使用 `nominal_fps`。
- 目标 FPS 可留空自动选择，且必须是正整数并不高于所有 episode、所有字段中的最低 FPS。
- 以所有字段公共时间区间的起点生成目标 FPS 时间轴；每个触发点分别选择各传感器时间戳最近的样本，距离相同时选择较早样本。
- 从 META 自动读取关节/EEF 分量名称、相机名称和图像 shape。
- 不完整或布局不一致的 episode 会跳过并在扫描结果中报告。

转换 job 创建时会一次性生成全部 segment 的 trajectory 清单。每个 `source_indices` 清单互不重叠且合起来覆盖全部 episode，子进程只接收自己 segment 的清单；后续动态启动仅用于限制并发和失败恢复，不会让 worker 争抢 trajectory。

该格式使用 Python pickle，只应加载 TeleAxis Collector 在本机生成的可信数据。相机引用会被限制在对应 episode 和相机目录内。

## 验证

```bash
/home/amin/miniconda3/envs/lerobot21/bin/python -m unittest -v
```

测试覆盖 HDF5 与 MultiProcessing Pool Dataset 扫描、字段映射、原始/输出预览、segment 写入、v2.1 合并、v3.0 打包、正常停止以及模拟异常 cache 的自动恢复。
