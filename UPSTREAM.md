# Baseline 来源与版本管理

## 固定来源

| 项目 | 值 |
| --- | --- |
| 上游仓库 | https://github.com/Hanchenli/vllm-continuum |
| Baseline 提交 | `316a58794a6ff86b216e579b74fd56ed0c5a911f` |
| 提交标题 | `Refactor README content for better readability` |
| 父仓库内路径 | `vllm-continuum/` |
| 使用的 Fork | 以 `.gitmodules` 中的 URL 为准 |

此提交标识研究采用的作者公开源码版本，不代表上游永远不变，也不表示
作者论文中的全部组件都已公开。实验方法应准确说明使用的版本和额外实现。

## 维护约定

1. 父仓库的 gitlink 固定子模块提交；普通 clone/update 不应自动升级 baseline。
2. Baseline Fork 的 `origin` 指向自己的 Fork，`upstream` 指向作者仓库。
3. 保留作者 Git 历史和许可证，不删除子模块 `.git` 或把作者历史重新初始化。
4. 自己的 baseline 改动放在 Fork 的独立分支；不要覆盖原始提交或改写已用于论文的历史。
5. 更新子模块版本前记录变更理由、代码差异、验证结果和受影响实验。
6. 对论文关键实验保存父仓库提交、子模块提交、实际运行代码版本、数据哈希和配置。
7. 如添加 baseline 标签，应使用新的明确名称，不强制移动已发布标签。

## 实际运行代码

H200-1 的现有 editable 安装使用独立副本：

```text
/export/home/ext.luohaowen1/continuum/workspaces/continuum-316a587
```

先前逐文件核查中，作者仓库的 3232 个已跟踪文件与此副本一致。
副本还含安装生成文件，不属于 baseline 源码改动。
该历史核查不是对未来运行副本状态的保证；每次正式实验仍应核实。
父仓库保留的独立研究实现可能通过继承、运行时接入等方式改变实际调度行为，
因此“子模块未改动”不能替代对实际实验方法的记录。
