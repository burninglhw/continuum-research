# 实验实现与解释边界

## 这次到底运行什么

这不是再次向DeepSeek付费请求，而是用已经采集的100题作为固定工作负载，让H200上的Llama-3.1-8B-Instruct执行真实prefill和逐token decode。Decode的采样结果被约束为原始可见响应的Llama token，模型前向与KV读写依然真实执行。这样下一轮固定prompt才可能复用上一轮真实计算出的相同token前缀。

全部3856个模型响应都会执行：3644个被接受响应、212个拒绝/恢复响应。两个没有得到响应的HTTP尝试单独登记，不能捏造其GPU decode成本。隐藏reasoning不进入Llama回放，这与DeepSeek原API总计算量不同。工具用采集的真实命令时长作等待事件；不重新启动SWE Docker，也不使用API响应耗时冒充本地推理耗时。最后一次提交工具的等待也计入JCT。

## 为什么不是“原封不动跑作者代码”

官方README明确写着without the estimation in the paper。公开代码只有两秒固定阈值策略，不含论文CDF估计器、PLAS、InferCept的完整实验实现。因此同时保留两个不同的Continuum对照：

- `continuum-public`：公开代码的两秒/历史均值策略，在共同的安全修复与监测层内运行。
- `continuum-paper`：按论文式(2)补建CDF TTL。`T`用最近128次因压力撤销保护/抢占后入队的实际等待时间均值近似；eta仅使用已经结束的job统计，并裁剪到[0,1]。新工具没有历史时使用公开代码的两秒冷启动值。这些是实现选择，不声称恢复了作者未公开细节。

vLLM基线使用同一个0.10.2 fork的FCFS路径，不伪称另一份未经修改的上游wheel。关闭cascade attention，因为原实现以refcount推断全batch公共前缀，而额外pin owner也会增加refcount。所有方法统一禁用，避免把错误KV计算当作加速。

另外共同修复：遍历pins时使用快照，避免删除时跳过下一项；返回请求成功获得APC引用后，才撤销旧请求owner的引用；补全抢占候选fallback。实际运行代码保存在独立研究目录，原始官方checkout、既有环境源码和100条trace不修改。

## ElasticKV版本选择

以optimized文档的连续前缀边界为主，采用motivation文档的admission-targeted reclamation：

1. 保留与重建Continuum相同的job级优先级、在线工具历史、TTL以及实际KV预算。
2. 只在原调度器的内存分配确实失败、达到其原有pin回收触发条件时回收，而不是凭预设显存水位主动删缓存。
3. 对每个合法边界计算近期返回概率乘实测插值的边际prefill节约，逐块撤销最低价值边界。
4. 每次重算实际分配缺口；共享引用/当前请求将要复用的块不能假装提供额外容量。
5. Unpin只减少owner引用。未被覆盖的suffix仍在APC哈希中，仍可命中。真正的物理eviction由原vLLM分配器触发，单独统计。

H固定为1秒，第一版边际价值没有额外queue-cost项。等大的候选block使用共同lambda时，lambda项不改变排序；本版由实际缺口约束停止，不声称验证了lambda动态定价贡献。Static对照用相同TTL，只固定25/50/75%保护；`conditional-whole`用同一个条件概率对whole对象排序，它不是文档中完整的周期性Dynamic-Continuum。CPU tiering、重新增加边界、CacheWise官方实现没有混进本版。

## 真实测量与估计的分界

- 每个run的JCT、实际prefix hit、prefill调度token、物理eviction、RAF和调度器耗时来自真实GPU回放。
- 图中RWS/PSR使用真实H200二维profile做单请求插值，是characterization估计；不是直接测了每个工具事件，也没有包含负载下的队列成本。
- profile随机token只用于测密集Llama计算形状，不用于替换100题的主实验输入。主实验使用所有原始prompt/response token。
- 时间到达由固定种子的Poisson interarrival产生；后续回合在上个真实模型响应完成后再等待对应工具时长。工具完成时刻被事件循环发现会有小量dispatch lag，逐请求保存。
- scheduler时间记录实际schedule路径，包含回收决策；额外结果汇总不计入该函数耗时，但其开销仍在端到端JCT中。
- historical-prefix recomputed tokens计的是本轮重新计算、且位于上一轮真实可复用LCP内的输入token；不是完整硬件FLOPs。prefill总token和物理eviction另存，避免混为一谈。
- pinned memory-time按唯一物理block去重；另分出工具仍在等待、非已经返回排队的protected memory-time。Llama BF16 KV为每token128KiB、16-token块2MiB。

## 通过的正确性检查

- 100题Llama token数与采集时记录逐响应一致，模板完成串严格以请求token串为前缀；最高总长度129382，未截断。
- 小型真实GPU测试：只保护512 tokens时，只要suffix未被覆盖，实际仍命中1040 tokens。
- 人为覆盖可回收块后，仅命中512-token受保护前缀；重新计算后原始logprob与冷启动相同。
- 未覆盖APC回放与冷启动的最大原始logprob偏差0.0232，小于预设0.05容差；这里比较的是强制采样之前的原始logprob，不能用强制输出相同作为数值正确性的替代。
- 一个owner取消保护后，另一个owner的共享引用和hash仍有效。
- 真实分配器缺32块时只撤销32块保护，RAF=1，未出现引用泄漏。
- 两题完整小试42个响应全部精确回放通过。小试不用于宣传100题性能收益。

## 哪些结论不能说

不能说复现了作者GPT-5的同一100题、A100/B200的原始倍数、70B/TP4、BFCL、LMCache offload、Autellix/InferCept或分布式500题实验。不能用强制输出回放评估SWE解题准确率，也不能把单个seed的差异称为稳定收益。若长上下文的100ms slack很小，只能说明低成本撤销大量长prefix的说法缺乏支持，不自动否定少量按缺口回收的收益；这需要以全量JCT和RAF结果判断。

所有未完成/失败运行保留原目录，不用删掉差结果后重复命名的方式制造正收益。
