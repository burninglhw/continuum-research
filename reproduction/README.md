# Continuum：小数据准备与分阶段复现

准备日期：2026-09-16。服务器：H200-1。

项目根目录：`/export/home/ext.luohaowen1/continuum`。

**当前完成的是数据准备和离线完整性检查，不是论文性能复现。没有下载模型权重、Docker 镜像，也没有启动 GPU 任务或付费 API 调用。**

## 1. 论文真正使用的数据

依据本机与服务器校验一致的 `论文/AriadneKV/Continuum-26ICLR.pdf`，第 3、7、8 页：

- 图 5、6 的主要性能实验使用作者用 GPT-5 收集的 SWE-bench 与 BFCL V4 Web Search 执行轨迹，每类约 100 条。轨迹包含多轮请求、工具返回及工具执行时间，不只是原始题目。
- BFCL 工作负载还有论文描述的 0.4 缩放、上下文筛选，以及 Poisson 到达过程。原始题目文件不包含这些处理后的轨迹。
- 图 7 是 500 道 SWE-bench Verified 题目的真实 agent 实验，而非仅回放问题文本。
- 模型是 Llama-3.1 8B/70B。论文中 8B 使用 A100/B200，70B 使用 H100/B200、张量并行度 4。当前机器是 H200，因此后续结果应标为 H200 上的复现/趋势验证，不能直接冒充原图数值。
- 论文基线为 vLLM 0.10.2；CPU KV offload 使用 LMCache 0.3.7。原始硬件对应的 offload 容量、chunked prefill、并发、模型上下文与到达过程都需要对齐。

当前固定的代码提交为 `316a58794a6ff86b216e579b74fd56ed0c5a911f`。该 checkout 的实验目录有作者的结果 JSON 和分析脚本，但没有找到上述两类完整的 GPT-5 回放轨迹及完整回放/缩放管线。**不能把这次新抽的 100 条题目当作作者的 100 条执行轨迹。**

## 2. 已准备的数据

以下路径均相对于项目根目录。

| 目录/文件 | 数量 | 用途 |
| --- | ---: | --- |
| `datasets/_sources/swe_bench_verified/data/test-00000-of-00001.parquet` | 500 条，2,096,679 字节 | 固定版本的官方 Verified test 原始文件 |
| `datasets/swe_bench_verified/data/test.jsonl` | 500 条，8,110,423 字节 | 无损转为本地 JSONL，保留完整官方字段 |
| `datasets/small/swe_bench_verified_100/test.jsonl` | 100 条 | 从 Verified 排序后用 seed=42 新抽样，非论文轨迹 |
| `datasets/smoke/swe_bench_verified_10/test.jsonl` | 10 条 | 上述 100 条中的前 10 条，用于冒烟检查 |
| `datasets/bfcl_v4_web_search/BFCL_v4_web_search.json` | 100 条独立问题，36,984 字节 | 论文涉及的 BFCL V4 Web Search 类别 |
| `datasets/bfcl_v4_web_search/possible_answer/BFCL_v4_web_search.json` | 100 条答案，75,962 字节 | 与问题 ID 一一对应的官方答案 |
| `datasets/bfcl_v4_web_search/multi_turn_func_doc/web_search.json` | 2,252 字节 | 官方工具定义 |
| `datasets/smoke/bfcl_v4_web_search_10/` | 10 条问题和对应答案 | seed=42 的新冒烟子集 |

数据文件总计 **12,391,985 字节，约 12.4 MB / 11.82 MiB**，不含少量说明、清单和脚本。这个数值包含本地转换与子集副本；实际选取的上游原文件总量约 2.25 MB。

BFCL 文档中的 Web Search 200 个测试场景，是 100 个基础问题分别在 snippet/no-snippet 两种模式下评测；不是本次漏掉了另 100 个独立问题。下载文件保持官方格式，扩展名 `.json` 的题目和答案实际为逐行 JSON。

保留了 SWE-bench dataset card、BFCL dataset card 和 BFCL 上游 LICENSE。SWE-bench 的标准答案补丁、测试补丁只用于评估；agent 提示词应按上游实现只读取 `problem_statement`，不要把答案补丁泄露给待测模型。

## 3. 固定版本与来源

- SWE-bench Verified：[官方数据集](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)，revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`，test split 500 条。
- 网络环境无法直连 Hugging Face，因此使用 [HF Mirror](https://hf-mirror.com) 获取该固定 revision；原始 parquet 按元数据记录的 LFS SHA-256 `a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd` 校验。清单同时保存官方 URL 和实际下载 URL。
- BFCL：[官方 Gorilla 仓库](https://github.com/ShishirPatil/gorilla/tree/7ad0134c665944819f88bc50862108d94015968b/berkeley-function-call-leaderboard/bfcl_eval/data)，revision `7ad0134c665944819f88bc50862108d94015968b`。这是 PDF 日期前的 2026-03-11 快照，不代表已获知作者采集时的精确版本。
- BFCL 文件通过 GitHub blob API 下载并检查 Git blob SHA-1；全部数据另计算 SHA-256。具体字节数、来源、派生关系、版本、随机种子在 `datasets/MANIFEST.json`。

没有下载 SWE-bench 全量训练集、其他 BFCL 类别、模型权重、任务环境镜像或网页缓存。下载脚本只取固定白名单，每个网络响应限制 10 MiB，避免误下大文件。

## 4. 立即可运行：纯 CPU 离线验证

不需要安装第三方包，不访问网络，不运行论文代码，不占用 GPU：

```bash
BASE=/export/home/ext.luohaowen1/continuum
python3 "$BASE/reproduction/verify_small_data.py" \
  --project-root "$BASE" \
  --report "$BASE/reproduction/validation_report.json"
```

检查项目：12 个数据文件的 SHA-256 与字节数、BFCL Git blob、记录字段、唯一 ID、问题答案对应关系、抽样可重复性、所有数据集数量。

如果官方代码 checkout 存在，还会复算作者已提交结果中的平均时延，并检查其 SWE-bench 实例 ID 都在当前 Verified 数据中。这只是**校验作者已公开结果**，不是重新执行性能实验。报告中的 `benchmark_executed` 始终明确为 `false`。

## 5. 下一阶段：先打通 10 条，再比较性能

以下是待执行流程，**本次没有执行**。先确认已安装官方 fork、mini-swe-agent 和 `datasets`，有可用的本地模型、获准使用的 GPU，并且目标任务镜像已就绪。SWE-bench 在镜像缺失时会自动拉取；数据集仅 2 MB 不代表整个实验也只占 2 MB。当前 `/export/home` 约 99% 使用率、可用约 76 GB，禁止直接批量拉取 500 个任务镜像。

### 5.1 本地数据如何接入

上游 `mini-swe-agent/src/minisweagent/run/extra/swebench.py` 允许 `--subset` 接收本地目录，内部调用 `load_dataset(path, split="test")`。已准备的目录包含可自动识别的 `test.jsonl`，无需再次从 Hugging Face 下载：

```python
from datasets import load_dataset

dataset = load_dataset(
    "/export/home/ext.luohaowen1/continuum/datasets/smoke/swe_bench_verified_10",
    split="test",
)
assert len(dataset) == 10
```

上面的加载示例需要额外安装 `datasets`；这次只执行了标准库验证，尚未安装推理环境。

### 5.2 未来的单 GPU 冒烟命令

这一步必须先确认 `MODEL_DIR` 是**已经存在的本地模型目录**、`GPU_ID` 是**已分配的 GPU**。8B 可用于打通流程，但 README 提醒较小模型运行真实 mini-swe-agent 可能失败，不能用 10 条 8B 冒烟结果替代 70B 的 pass rate 或论文图 7。

下面的单 GPU 示例只用于 8B；70B 应另行规划 4 卡资源。为避免覆盖旧结果，每次创建唯一运行目录。先在同一 fork 中切换 `fcfs` 和 `continuum` 做功能对比；严格复现时还需单独对齐原版 vLLM 0.10.2 的基线。

```bash
BASE=/export/home/ext.luohaowen1/continuum
: "${MODEL_DIR:?请先设置本地 Llama-3.1-8B-Instruct 模型目录}"
: "${GPU_ID:?请先设置获准使用的 GPU 编号}"
test -d "$MODEL_DIR"
MODE=fcfs
RUN_DIR="$BASE/reproduction/runs/$(date +%Y%m%d_%H%M%S)_${MODE}_smoke"
mkdir -p "$RUN_DIR/agent"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$BASE/reproduction/cache/huggingface"
RUN_OUTPUT_DIR="$RUN_DIR" CUDA_VISIBLE_DEVICES="$GPU_ID" \
  vllm serve "$MODEL_DIR" \
  --served-model-name continuum-local \
  --scheduling-policy "$MODE" \
  --tensor-parallel-size 1 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 2048 \
  --host 127.0.0.1 --port 8100
```

在另一个终端设置同样的 `BASE`、刚才实际生成的 `RUN_DIR` 和离线/cache 环境变量，然后在服务就绪且所有选中任务镜像已就绪后运行：

```bash
mini-extra swebench \
  --model-class vllm \
  --model continuum-local \
  --subset "$BASE/datasets/smoke/swe_bench_verified_10" \
  --split test \
  --workers 1 \
  --output "$RUN_DIR/agent"
```

此处刻意不传 agent 的 `--port`：当前 checkout 把该选项直接写成 `model.port`，而 `VllmModelConfig` 没有该字段；直接照 README 传入可能报错。先使用客户端默认的 `http://localhost:8100/v1`。这一兼容性问题没有修改到作者代码中。

完成后正常停止自己启动的服务，确认 `RUN_OUTPUT_DIR` 下存在 `scheduler_timestamps`，再分析该次日志，不要覆盖作者随仓库提交的参考结果：

```bash
python "$BASE/vllm-continuum/continuum_exp/analyze.py" \
  --input-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR/metrics"
```

10 条、单 worker 只能验证流程，无法验证调度收益。第二次使用全新的 `RUN_DIR`，把 `MODE` 改为 `continuum`，保持其他设置一致。之后再扩展到 100 条、受控并发/Poisson 到达，记录平均 job completion time、P95、吞吐量、失败率与任务通过率；至少多次重复。当前上游 `--use-jps` 直接调用未固定 seed 的 `random.expovariate`，公平 A/B 前还需固定同一到达时间序列，不能仅设相同 JPS 就声称两次负载完全相同。

## 6. 完整对齐论文仍需的材料

1. 作者的 SWE-bench/BFCL GPT-5 完整轨迹、精确采样 ID、工具调用时长、网页/工具返回内容、回放程序及 BFCL 0.4 缩放实现。应先向作者索取，不能通过下载原始题目自动得到；自行采集属于新工作负载，还会产生 API/浏览成本。
2. 合法获准使用的 Llama-3.1 8B/70B 权重及本地路径。按 BF16 参数量粗估，8B 权重约 16 GB，70B 约 140 GB，均不在本次“小数据”下载范围。70B 尤其超出现有约 76 GB 的剩余磁盘空间；实际分片/附加文件大小仍需另查。
3. 固定版本的推理和评测依赖、可分配的 GPU、必要的任务镜像和工具网络权限，以及 H200 上的硬件相关成本模型校准。BFCL 的工具定义本身不会执行网页搜索，也不等于已准备了完整 evaluator/搜索服务。
4. 对齐模型、tokenizer、上下文限制、采样参数、内存比例、chunk size、CPU offload、任务到达序列和并发压力。H200 的绝对时延不能直接与论文中的 A100/B200/H100 数值比较。

## 7. 重新准备数据

`prepare_small_data.py` 是本次实际使用的准备程序，依赖版本见 `requirements-data.txt`。验证不需要这些依赖。若将来确有必要重新下载，应在独立环境中安装数据工具，再运行准备程序。默认使用可达的 HF Mirror；官方服务可访问时可改为 `--hf-endpoint https://huggingface.co`。

```bash
python reproduction/prepare_small_data.py \
  --project-root /export/home/ext.luohaowen1/continuum \
  --hf-endpoint https://huggingface.co
```

准备程序拒绝覆盖内容不同的既有数据文件；生成的 manifest 记录实际下载来源。没有修改 `vllm-continuum` 工作区或原论文。
