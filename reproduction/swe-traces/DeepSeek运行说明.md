# DeepSeek SWE-bench trace

接口为 `https://api.deepseek.com/chat/completions`，固定模型为 `deepseek-flash`。使用 `thinking.type=enabled`、`reasoning_effort=high`，流式响应保留回答和 `reasoning_content`，单次输出上限为 16,384 tokens。

`collect.py --config deepseek_config.json` 选择独立配置。新运行保存到 `runs/deepseek-flash-100-20260918`。之前的配置、运行结果和费用账本保留，修改前脚本在 `backups/before-deepseek-20260918`。

先运行固定清单的前三题，检查完整请求、响应、真实工具调用、提交和 patch；不把提交等同于 SWE-bench 修复测试通过。`run_deepseek.sh` 通过标准输入接收密钥，不将密钥写入配置、命令行参数或日志。

2026-09-18 官方账户查询显示 10 CNY 余额。新平台的软件支出上限为 9 CNY，之前平台的保守估计 33.5589625 CNY 计入合并账本，总授权仍为 300 CNY。计费使用高峰输入 2 CNY/百万、输出 8 CNY/百万，缓存不打折；该估计不等于平台实际账单。

本采集器使用文本 bash 动作，不发送 API 的 `tools` 参数。按 DeepSeek 官方文档，前一轮 `reasoning_content` 不进入后一轮上下文。记录中将可见回答 token、思考 token、API 计费 token 分开保存。回放实验应显式选择是否模拟思考生成长度，不能把这些口径混用。

文档入口：

- 首次调用：https://api-docs.deepseek.com/zh-cn/
- Chat Completions：https://api-docs.deepseek.com/zh-cn/api/create-chat-completion
- 多轮对话：https://api-docs.deepseek.com/zh-cn/guides/multi_round_chat
- 思考模式：https://api-docs.deepseek.com/zh-cn/guides/thinking_mode
- 模型与价格：https://api-docs.deepseek.com/zh-cn/quick_start/pricing

基本命令：

```bash
cd /export/home/ext.luohaowen1/continuum/reproduction/swe-traces
bash run_deepseek.sh
python summarize.py --run-dir runs/deepseek-flash-100-20260918
python quality.py --run-dir runs/deepseek-flash-100-20260918
```

宿主机磁盘预留、Docker 容器隔离、单 worker、固定任务清单、Llama 128K 回放上下文检查仍启用。新接口默认不自动重试；若失败，先查看错误和账本再恢复。
