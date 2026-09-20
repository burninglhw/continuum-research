# Continuum research workspace

用于保存论文 baseline、复现工具及独立研究实现的源码工作区。
本仓库不包含服务器环境、完整数据集、实验轨迹、模型权重或安装缓存。

## Baseline 固定版本

- 上游：<https://github.com/Hanchenli/vllm-continuum>
- 论文 baseline 提交：`316a58794a6ff86b216e579b74fd56ed0c5a911f`
- Baseline 路径：`vllm-continuum/`，使用 Git 子模块管理。
- 子模块地址以 `.gitmodules` 为准；父仓库提交记录精确的子模块提交。
- 该版本为作者公开实现，不应表述为包含论文全部估计器的实现。

保持 baseline 源码不变；新的算法、接入代码和分析首先放在独立研究目录。
如需要修改 baseline，应在自己的 Fork 新建分支，并记录对应实验使用的提交。
不要使用 `git submodule update --remote` 隐式升级论文 baseline。
更多来源和维护约定见 [UPSTREAM.md](UPSTREAM.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `vllm-continuum/` | 固定版本的 Continuum/vLLM baseline 子模块 |
| `reproduction/conda-setup/` | 环境构建脚本与依赖清单 |
| `reproduction/swe-traces/` | SWE 轨迹采集、协议检查、导出和测试 |
| `reproduction/swe-own-image/` | SWE 任务 Docker 镜像配方、构建和验证 |
| `reproduction/catraces-cpu-20260919/` | 离线轨迹转换和合成负载工具 |
| `elastic-kv-study/pilot-20260917/` | 前期数据处理与分析脚本 |
| `elastic-kv-study/swe100-20260920/` | 调度策略、GPU 回放、验证与结果分析源码 |

## 获取源码

使用 GitHub 页面上的仓库克隆地址，并添加 `--recurse-submodules` 参数。
如果已经克隆父仓库，在仓库根目录执行：

```bash
git submodule update --init --recursive
git -C vllm-continuum rev-parse HEAD
```

首次版本的第二条命令应输出上面的 baseline 提交。
后续复现时，应以所复现的父仓库提交记录的子模块版本为准。

## 环境与实验边界

首次纳入的内容是源码快照，不是开箱即用的完整实验备份。
现有脚本有 H200-1 的绝对路径和历史运行路径；移植前需要核对路径、依赖、
模型/数据来源及本地配置。不要直接执行历史恢复或迁移脚本。

H200-1 当前安装环境的 editable 源码位于：

```text
/export/home/ext.luohaowen1/continuum/workspaces/continuum-316a587
```

它是 baseline 的独立运行副本，不是子模块 checkout 本身。
修改或切换子模块不会自动改变已安装环境；本次建仓没有重装环境或切换运行代码。
以后若调整运行副本，必须同时核实 import 来源并记录实验版本。

`.gitignore` 使用保守白名单。环境、缓存、数据、轨迹、日志、生成结果、
论文 PDF、未公开构思及运行配置默认不纳入。
新增源码模块或脱敏配置示例需要显式更新白名单；首次版本没有上传 JSON 配置。
正式复现还需单独准备授权的数据、配置和模型，并记录版本、来源与校验和。

## 发布与许可

本父仓库为公开的研究源码仓库，提交内容及历史可被所有人查看和克隆。
每次上传前应审查代码、文档、配置示例、数据许可、内部信息及凭据。
公开可见性不等于获得第三方材料的再分发授权，也不自动赋予额外开源许可。
不得提交 API key、Token、私钥、环境目录或完整请求日志。

保留子模块原有 LICENSE、NOTICE 和作者来源；不要将上游代码标注为个人原创。
本仓库的建立不为独立研究代码新增开源许可证。
