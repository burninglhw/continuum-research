# CATraces → Continuum：CPU 数据准备执行结果

## 当前状态

已完成数据转换、严格子集筛选、合成 token 配方和 CPU 一致性验证。**没有启动 GPU，没有调用付费 API，没有运行 SWE-bench 任务容器，也没有生成性能收益数据。**

这些是脱敏日志驱动的回放材料，不是原始 Claude 文本、不是 SWE-bench 的 100 道新题，也不是 Continuum 原图的严格复现。

## 1. 实际得到的数据

| 项目 | 本次结果 |
|---|---:|
| 冻结的原始会话文件 | 94 |
| 预先固定的 train / test 会话文件 | 75 / 19 |
| 全部 assistant 记录 | 20,634，逐条且仅一次映射 |
| 按 request、message、边界联合归组 | 11,509 组，包含隔离组 |
| 满足当前 CPU pilot 筛选条件的 episode | **328** |
| train / test 可用 episode | **291 / 37** |
| train / test 可用 episode 来源会话 | 49 / 11 |
| 可用 episode 总轮数 | 1,938 |
| 选定首轮 pilot | **8 个 episode，来自 8 个不同测试会话** |
| pilot 总轮数 / 工具等待 transition | **67 / 59** |
| 前缀假设 | 0% 复用、50% 复用、append-only 高复用 |
| 三种前缀场景的 token 配方 | 201 条 |
| 实际在 CPU 上生成并检查的 token IDs | 18,592,275 |

11,509 不是替代原始事件计数的“真实推理总次数”。它比初次审核的 11,498 组多，是因为不再把不同 message 或不同 episode 的同名 requestId 强行合并；有冲突的组仍被隔离。

episode 是父子链中从可识别 user 输入到无工具的末尾 assistant 文本回复的一段，不表示问题已正确解决，也不能认证日志记录了最后一个 decode token。结束证据和不确定性均保留在输出中。

## 2. 筛选和分组规则

固定 seed 为 `20260919`。先对全部 94 个源会话做哈希排序并固定 75/19 分组，再转换、筛选 episode；不根据实验收益重新分组。

可用子集要求：

- 主链完整可追踪，不能把子代理 sidechain 或父链分叉硬串起来。
- 2–20 轮；除最后一轮外，每轮恰好一个工具，且结果位于正确的父子链上。
- 最后一轮有文本回复、无未完成工具；遇到 API 错误、身份冲突、缺失依赖等隔离。
- 每轮 prompt+output ≤131,072；所有保留的长度按日志报告值，不静默截断。
- 每次工具观测延迟 ≤60 秒；超限整个 episode 不进入 pilot 子集，不修改时长。
- Task、ExitPlanMode、AskUserQuestion 等嵌套代理/交互工具暂不进入 pilot。
- compaction 的 summary、transcript-only 记录不会误算成新的人类输入。

总共划分出 2,754 个图边界组件：328 个 accepted、944 个 quarantine、1,482 个 excluded。**这些不是 2,754 个独立原始会话**，其中 1,164 个组件没有 assistant 请求。`episode_decisions.jsonl` 保存每个组件的决定和原因。

- `quarantine`：身份/因果/结束状态不满足当前严格回放条件，或属于暂不支持的分支类型；不一定是原数据本身损坏。
- `excluded`：不满足预声明的 pilot 范围，例如多工具、上下文超限、工具过长、轮数不符。
- 同一个组件可能有多个原因，不能把各原因计数直接相加当成组件总数。

这是较严格的小规模子集，存在选择偏差。不能拿这一子集的结果代表 CATraces 全量长尾和复杂并发情况。

## 3. 首轮 pilot 的固定样本

选择按另一个固定哈希顺序，在测试会话之间选 8 个不同来源，各取一个 episode；没有按估计收益挑选。

| 源会话 / 起点节点 | 轮数 | prompt 范围 | 观测工具等待累计 |
|---|---:|---:|---:|
| project_007/session_0079 / node_00314 | 2 | 91,508–92,367 | 1.991 s |
| project_006/session_0029 / node_00847 | 10 | 115,927–126,092 | 34.106 s |
| project_006/session_0023 / node_00265 | 3 | 81,773–82,885 | 47.473 s |
| project_005/session_0014 / node_00141 | 2 | 79,827–80,190 | 39.750 s |
| project_006/session_0047 / node_00033 | 13 | 24,181–73,093 | 43.192 s |
| project_006/session_0039 / node_00126 | 20 | 71,233–101,668 | 42.534 s |
| project_001/session_0001 / node_00264 | 3 | 125,005–125,929 | 29.517 s |
| project_006/session_0048 / node_00401 | 14 | 111,812–123,663 | 24.806 s |

表中等待是源日志的观测延迟总和，不是 H200 运行耗时或 JCT。首轮可计划并发 1/2/4/8，但目前尚未开始这些 GPU 实验。

## 4. 前缀及目标模型

只读检查了 H200-1 已下载的 `LLM-Research/Meta-Llama-3.1-8B-Instruct` tokenizer 和 config。没有加载模型权重。

- `max_position_embeddings = 131072`，`vocab_size = 128256`。
- 合成 token 使用 `[256, 32000)`，已验证全部 ID 存在于 tokenizer 的基本词表中，且不属于 special tokens。
- tokenizer SHA256：`79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4`。
- 采用 SHAKE256 的确定性 token 生成，保存输入/输出 token 哈希、保留前缀长度和重置原因。
- 高复用场景保留上一轮输入及输出；放不下完整上一轮历史时显式重置，不假装知道真实 compaction 行为。
- 每种场景都检查逐轮 LCP、独立 episode 的首个 16-token 缓存块隔离。
- 三种场景共享每个请求的固定合成输出序列；同一场景供所有策略共用相同配方。

配方文件很小，可以随时重建 token IDs，因此未保存数十 MB 的重复 token 数组。`synthesize.py` 中的 `materialize_episode()` 是重建入口。

**仍然不能认证的事项**：原始真实公共前缀、Claude 文本在 Llama tokenizer 下的真实长度、源日志最终完整 output token 数。328 个可用 episode 的 1,938 轮中仍有 1,920 轮缺少 stop_reason，406 轮报告的 output_tokens ≤2；这些已标记为 as-reported，不进行无依据的补长。

## 5. 验证结果

本地 `validation_report.json` 为 **PASS**：

- 35,127 条源图记录、20,634 条 assistant 记录均恰好一次映射，无静默丢弃。
- 检查每个请求合并后的工具集合与原始分块一致。
- 全量源数据有 11,617 个 tool_use 块；去重后的 `(源文件, tool_id)` 为 11,533 个。某些 ID 有重复/跨请求复用，已隔离；它们都不是已经认证的独立工具执行次数。
- 对 328 个接受 episode 重新沿原始 parentUuid 逆向遍历，确认主链、工具因果、时间顺序、长度和延迟。
- 预测历史只含 accepted train episode 的工具样本，测试集没有进入初始历史。
- `scheduler_metadata.jsonl` 只含当前响应完成时公开的工具名、episode/turn 身份和终止标记，未来等待仅保存在 controller 文件。
- 14 项边界检查通过，包括 request/message 冲突、多工具分块合并、未知结果、compaction、分叉、环、超长工具、非法 usage、子链、前缀重置。
- 在第二目录重新执行转换和生成，产物逐字节一致；token 配方再次完整重建并检查。

这验证的是**离线数据与接口约定**，不是运行中的服务端信息隔离，也不是 GPU 调度器正确性；服务器适配器尚未实现。

## 6. 文件说明

脚本均只依赖 Python 标准库，无需为了数据准备再安装 Torch/vLLM 或建立新模型环境。

| 文件 | 用途 |
|---|---|
| `convert.py` | 按父子图切段、合并请求工具块、隔离异常、固定训练测试与 pilot |
| `synthesize.py` | 生成三种合成前缀配方、分离 scheduler 与 controller 元数据 |
| `validate.py` | 全量源记录覆盖、因果关系、14 项边界条件、重复运行一致性验证 |
| `cpu-run/accepted_episodes.jsonl` | 328 个接受 episode，带逐轮原始来源 |
| `cpu-run/quarantine_episodes.jsonl` | 隔离组件及原因 |
| `cpu-run/excluded_episodes.jsonl` | 范围外组件及原因 |
| `cpu-run/pilot_episodes.jsonl` | 冻结的 8 个测试 episode |
| `cpu-run/requests.jsonl` | 所有请求分组及质量标记，含不可回放的组 |
| `cpu-run/split_manifest.json` | 原始会话级 train/test 分组，不可事后重洗 |
| `cpu-run/training_history.json` | 仅训练集的观测工具延迟 |
| `cpu-run/token_recipes.jsonl` | 三种场景的 201 条 token 配方及哈希 |
| `cpu-run/scheduler_metadata.jsonl` | 只允许按当前响应事件交给调度器的信息 |
| `cpu-run/controller_transitions.jsonl` | 回放器专用的未来等待，不得交给估计器 |
| `cpu-run/validation_report.json` | CPU 验证结果 |

## 7. H200-1 部署目标和复跑方法

本任务的独立目录为：

`/export/home/ext.luohaowen1/continuum/reproduction/catraces-cpu-20260919`

不改动原 `vllm-continuum` checkout，也不覆盖旧 SWE-bench 采集或费用账本。源数据和 tree 元数据随独立 CPU bundle 保存。

在上述目录内，CPU 复跑命令为：

```bash
export PYTHONDONTWRITEBYTECODE=1
SOURCE=data/cachewise-coding-traces-181c435a090d328d00bbbee4c8eeb27d32f3abd2
python3 convert.py --source "$SOURCE" --tree data/source-tree.json --output cpu-run
python3 synthesize.py --run cpu-run --token-domain cpu-run/token_domain.json
python3 validate.py --source "$SOURCE" --tree data/source-tree.json \
  --run cpu-run --repeat-dir work/repeat-check
```

这组命令不会启动 GPU。这里的 CPU 元数据 `ready` 不代表可直接运行 `vllm serve`，所有请求仍带 `gpu_replay_ready=false`。

## 8. 下一阶段仍需完成

1. 实现固定 token/输出回放与当前工具名的服务端桥接，验证原 fork 的 pin/unpin 真正触发。当前随机文本不能依赖 bash parser 识别工具。
2. 先比较相同代码基座下的 FCFS+prefix cache 与 Continuum-release 的均值判断/2 秒 pin；论文完整 CDF/成本估计器另行实现、验证与标注。
3. 确认 GPU 使用条件后才运行 1/2/4/8 并发，交错重复，报告 E2E JCT、serving-only latency、TTFT 和重算/KV 指标。
4. 后续扩大样本并报告筛选偏差、不同前缀假设和长尾影响；在 GPU 测量完成前不绘制“性能收益”图。
