# LeRobot Data Convert

本地多进程 LeRobot 数据转换工作台。后端使用 Python、SQLite 和原子 JSON manifest，前端是无构建步骤的 PWA。

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

CPU 滑杆决定 worker 数以及可用 CPU 集，每个 worker 固定到一个核心。内存预算结合适配器给出的单 worker 估算限制并发数，任务详情显示实际 worker 数和实时 RSS。

## 原始数据适配器

任何格式只需实现 [`RawDatasetAdapter`](./lerobot_dataconvert/adapters.py) 的三个方法：

```python
from lerobot_dataconvert.adapters import RawDatasetAdapter, register_adapter


@register_adapter
class MyAdapter(RawDatasetAdapter):
    slug = "my_format"
    display_name = "My Format"
    description = "My robot recording layout."

    def inspect(self):
        # Return DatasetDescriptor with stable EpisodeRef entries.
        ...

    def iter_frames(self, episode):
        # Yield FrameSample(state, action, {camera_name: rgb_uint8_hwc}).
        ...

    def preview(self, episode, camera, frame_index):
        # Return one RGB uint8 HWC image.
        ...
```

Cache、进程调度、资源限制、LeRobot 写入、revision、合并、ETA 和 UI 不需要在适配器中实现。

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

## 验证

```bash
/home/amin/miniconda3/envs/lerobot21/bin/python -m unittest -v
```

测试覆盖 HDF5 扫描与预览、segment 写入、v2.1 合并、v3.0 打包、输出视频预览、正常停止以及模拟异常 cache 的自动恢复。
