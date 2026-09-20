# H200-1：100 条 SWE-bench trace 采集准备

## 当前结论

截至 2026-09-18，本次已创建独立 Conda 环境、锁定 100 个任务、实现采集及统计脚本，并完成自建基础镜像、共享依赖层和第一题环境验收。24 项安全/计费/本地镜像检查通过，第一题 Django 测试执行 101 项（99 项通过、2 项跳过），题间工作区隔离验证通过。

**目前 SWE-bench trace 数为 0：第一题的自建环境已就绪，其余任务环境尚待逐组构建与验收，未开始付费任务采集。** 之前一次简短 API 冒烟生成成功，不属于 SWE-bench trace。没有生成或伪造性能对比图，没有启动 GPU 推理，没有创建后台采集或定时监控任务。

最初的镜像下载问题与当前解决路径：

- H200-1 通过 Docker daemon 拉取 `docker.io/swebench/sweb.eval.x86_64.django_1776_django-13413:latest` 时，访问 Docker Hub 超时。
- 服务器已有 DaoCloud 镜像站明确拒绝该任务镜像：不在其 allowlist。
- 未更改系统 Docker、网络、代理、DNS 或安全设置；未绕过镜像站的 allowlist。
- 根据用户不复用他人镜像的要求，已改用 Canonical 官方 Ubuntu Base 根文件系统，核对 GPG 签名及 SHA-256，独立构建自己的镜像，不依赖 Docker Hub。
- 固定 100 题的环境配方可分为 27 个候选共享依赖组；第一个组含 8 题，目前只验收了 `django__django-13413`，不能当作 100 个任务环境均已就绪。

具体镜像名称、共享空间、来源、验证结果及操作见同目录《自建镜像说明.md》。后续按依赖组逐批构建、复用，避免一次占满共享盘。原 Continuum checkout 保持干净。

## 模型与预算

- 用户总预算：**人民币 300 元**。
- 用户已接受替代/混合模型，因此本批不标作原论文 GPT-5 精确复现。
- 当前请求模型名：`claude-sonnet-4-6`。
- 成功冒烟请求的平台返回模型名：`gpt-5.5`。这是网关返回值，不是独立验证的真实上游身份证明。
- 先前直接请求 `gpt-5.5` 返回 HTTP 503 / `model_not_found`；平台指出该令牌属于 `Claude Code CX`，没有该名称对应的可用通道。失败记录保留，不会被覆盖为成功。
- 成功冒烟的 usage：prompt 118、completion 5、total 123 tokens。没有传送研究 idea、私人代码或密钥给模型；正式采集仅发送公开任务内容和隔离任务仓库的观察结果。
- 脚本按输入 17.5、输出 105 元/百万 token 的保守上界估算，缓存输入不打折，以 270 元为软件停止阈值，为 300 元预算留余量。每题另有 8 元保守阈值。
- 每次请求前预留输入/输出上界费用；缺失或异常 usage、接口错误、超时都会停止，不自动重试或替换模型。无法确定收费的失败请求按预留上界计入安全账本，不报告成确定的实际账单。
- **这不是平台强制额度，真实费用以平台账单为准。** 建议在平台给专用令牌同时设置不超过 300 元的实际额度；不要将无限额度误当免费。预算可能在完成 100 个任务前用完，届时必须停止，而不是悄悄提高预算或补造数据。
- 密钥只在输入和进程内存中使用，不写入配置、shell 命令参数、trace 或普通日志；程序退出后重新输入。不在本说明中保存密钥。

## Conda 环境

服务器目录：

```text
/export/home/ext.luohaowen1/continuum/envs/continuum-traces
/export/home/ext.luohaowen1/continuum/reproduction/swe-traces
```

在 H200-1 的 Bash 终端激活：

```bash
source /export/home/ext.luohaowen1/continuum/tooling/miniforge3/etc/profile.d/conda.sh
conda activate /export/home/ext.luohaowen1/continuum/envs/continuum-traces
```

也可以：

```bash
source /export/home/ext.luohaowen1/continuum/reproduction/swe-traces/activate.sh
```

已安装 Python 3.11.16、作者 fork 自带的 mini-swe-agent 1.14.4、SWE-bench 4.1.0、NumPy 1.26.4、Transformers 4.55.2、Tokenizers 0.21.4、requests 2.32.5 等。完整版本见同目录 `pip-freeze.txt` 和 `conda-explicit.txt`。

采集环境没有安装 PyTorch/vLLM，也不需要加载 GPU 模型。Transformers 提示“没有 PyTorch、只能用 tokenizer”等属于预期。原 Continuum 推理环境及原代码 checkout 保持不变。

Conda 管理采集程序的 Python 依赖，Docker 隔离模型生成的 shell 命令，两者用途不同。**不会把模型生成的命令放在 H200-1 宿主机或仅靠 Conda 来执行。**

## 与论文对齐的边界

用户提供的 `Continuum-26ICLR.pdf` 第 3 页表 1、§2.2 和第 7 页 §4.1 给出：

| SWE-bench 指标 | 论文均值 | 论文标准差 | 本次处理 |
|---|---:|---:|---|
| 每程序轮次 | 10.9 | 2.1 | 实测，不固定成 11 轮 |
| 工具执行时间 | 925 ms | 3,550 ms | 容器内部单调时钟计时，另记 Docker 往返耗时 |
| Token per program | 70,126 | 19,732 | 同时记录 API 累计输入/输出、Llama 累计请求和最终上下文口径 |

这些是采集结果的分布统计，不是作者公布的每题硬性限制。论文没有清楚公布这 100 题的 ID、精确 GPT 快照和 token 聚合定义；不能把 70,126 直接等同于最后一轮上下文长度或完整账单。

本次沿用作者 fork 的 mini-swe-agent SWE-bench 提示词和 10,000 字符工具观察截断逻辑，同时保存完整原始工具输出及模型实际看到的观察。50 步、900 秒/题、60 秒/工具属于工程安全上限，**不是表 1 的目标统计量**。不添加 sleep、不补 token、不强截轨迹来凑表中的数字。

使用已下载 Llama-3.1-8B-Instruct tokenizer 做离线长度统计和 128K 回放上下文检查，不加载其权重。GPT 网关计费 token 与 Llama 回放 token 分开记录。失败、到达限制、未提交、未经过 SWE-bench 测试验证的任务都明确保留其状态；`Submitted` 不等于题目修复测试通过。

## 100 个任务的来源

- 数据集：`princeton-nlp/SWE-bench` 原版 `test`，2,294 题。
- 固定数据 revision：`e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6`。
- 按 `instance_id` 排序后，`random.Random(42).sample(..., 100)` 一次性选定，不按实验结果挑题或替换失败题。
- 前三题：`django__django-13413`、`django__django-10426`、`matplotlib__matplotlib-26472`。
- 这不是 SWE-bench Verified 100 题，也不是作者公开的原始 100 题列表。
- 任务输入剔除 gold patch、test patch、参考答案和 hints。清单及 SHA-256 在 `dataset-manifest.json` 和 `tasks-100.jsonl`。

## 使用自建镜像的执行顺序

先查看自己的镜像，并执行不收费的环境检查：

```bash
df -h /export/home
docker image ls --filter 'reference=continuum-luohaowen/*'
cd /export/home/ext.luohaowen1/continuum/reproduction/swe-traces
source activate.sh
python collect.py --limit 1 --check-images-only
```

不要执行全局 `docker system prune`，不要删除其他人的镜像或容器。构建程序开始时要求至少 12 GiB 空闲，构建中保留 8 GiB。采集器已禁用自动拉取镜像，只接受验收清单中的固定自建 image ID。

当前第一题可单独开始付费生成；下面仅提供命令，本次未执行：

```bash
cd /export/home/ext.luohaowen1/continuum/reproduction/swe-traces
python collect.py --limit 1
python summarize.py
```

密钥在交互提示中隐藏输入，不要放进命令行。第二题 Python 3.5/Django 和第三题 Matplotlib 环境尚未就绪，因此当前直接执行 `--limit 3` 或 `--limit 100` 会在输入密钥前停止。先逐组补齐对应环境并通过验收，再使用同一固定清单扩大规模：

```bash
python collect.py --limit 100
python summarize.py
```

可以在用户自行启动的 `tmux` 会话中运行以避免 SSH 断开终止。当前没有启动该会话或任何后台收集。

程序只允许单 worker；并发执行由运行锁阻止。已形成 trace 的任务不会重复计费执行。遇到已有零响应失败任务会要求检查，不会静默跳过它来声称完成 100 条。程序或参数改变后不要把新旧配置的结果混在同一次运行中。

## 产物与安全验证

运行目录：

```text
/export/home/ext.luohaowen1/continuum/reproduction/swe-traces/runs/nowcoding-mixed-100-20260918
```

每题将保存：

- `task.json`、`image.json`、`baseline.json`：任务、镜像 digest/ID 和基线来源。
- `requests.jsonl`、`responses.jsonl`：准确请求上下文、请求/返回模型、API usage 和延迟，不含鉴权头。
- `tools.jsonl`、`messages.jsonl`：原始工具输出、内部计时、Docker 墙钟耗时、模型实际观察。
- `trajectory.json`、`patch.diff`、`metrics.json`：完整可见轨迹、修复补丁与状态。
- 运行级 `budget.json`、`uncertain-costs.jsonl`、`summary.json`、`metrics.csv`。

隔离目标：无 GPU、无主机目录绑定、无 Docker socket、无 API 密钥转发；模型执行阶段网络关闭，根文件系统只读，任务工作区放在限额 tmpfs 中。默认每个任务 4 CPU、8 GiB 内存、256 PID、3 GiB 工作区。工作区限制或依赖不满足会停止并报告，不能冒充成功实验。

已通过：24 项单元检查、`pip check`、自建第一题环境测试和真实工作区隔离验证。早期已有镜像计时器测试保留为历史记录，不再作为当前任务镜像来源。**环境测试不是模型采集 trace，也不是 SWE-bench 修复通过率评测。**

经用户批准，之前仅删除重复文件 `original/consolidated.00.pth`，释放约 16.06 GB；四个 safetensors 分片和索引保留。本次没有删除模型、其他用户镜像或容器。自建镜像链共享底层，合计约 2.02 GB，另有下载包/元数据开销；检查时共享盘约剩 20.9 GB，其他用户写入会改变余量。
