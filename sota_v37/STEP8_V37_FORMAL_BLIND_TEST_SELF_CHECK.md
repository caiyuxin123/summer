# SOTA步骤8：v3.7 正式 February 盲测与前序自检报告（2026-09-30）

## 1. v3.6 正式测试前自检：发现并修正的问题

进入 February 正式目标前重新核查 v3.6，发现两个不能直接放行的问题。

1. `model_lock_v36.json` 中记录的 protocol SHA256 与最终协议文件不一致。原因是协议说明在锁参之后又补充了正式评价条款。已在正式目标接入前冻结 v3.6.2 协议，并让 lock 重新绑定协议、grid、calibration、hidden-state 和 common projection 的真实 SHA256。
2. v3.6 初版在 96/24 校准验证中曾把真值 `state_active` 用于候选裁剪和误差 mask。该量属于校准隐状态标签，正式 February 粗观测中不存在，因此会形成校准验证与正式推理不一致。v3.6.2 已完全禁止 hidden activity mask：验证预测、指标、全量重训和正式推理均不使用该量，状态 MAE 在全部 168×48 通道上计算。
3. 补充了 package portability 修复，使 `pytest -q` 与 `python -m pytest -q` 都可直接通过；最终正式目标前测试为 11 passed。

修正后四方法的锁定超参数未发生变化：SF* 为 PCA=8/alpha=0.1，PM* 为 PCA=16/alpha=1.0，PM-DSE 为 harmonics=2/alpha=0.1，PI-GNN 为 hidden=32/lr=0.003/120 epochs/wd=1e-4。由于 activity-mask 规则改变，验证 profile 指标重新计算并冻结；不再沿用 v3.6 初版指标。

## 2. Phase A：只接入正式 target，不接入 Reference

正式 `target_input_only.npz` 的 SHA256 为 `15b5857aed7e507149e7035381cdc5f44238c2768428fa65411dacbf12891dde`，键严格为 `graph/x_history/x_month`。Phase A 工作区不存在 `reference_only.npz`。

SF*、PM*、PM-DSE 在全部 120 个目标期外校准场景上按锁定超参数各重训一次；PI-GNN 使用正式冻结的 10 个种子 `20260929—20260938` 各重训一次。13 个候选细粒度状态随后全部通过 byte-identical 的公共物理投影和同一 CEF 后处理。所有候选状态和 projected monthly predictions 先冻结，之后生成：

`PHASE_A_BLIND_FREEZE_MANIFEST.json`

其 SHA256 为 `531b4c1ed7cf59380602b62448a8ed1b48db221c50616b256c06cdd708530374`。在该 manifest 形成时，Reference 仍未接入。

## 3. Phase B：一次性接入 Reference

Phase A 冻结后才接入 `reference_only.npz`，其 SHA256 为 `74ace50c9de4fa577f36096e716fb7999f48e4e9b68b78677db762f34bc32cdb`，文件只含 `y`。评价程序首先重新验证全部 Phase-A projected 文件 SHA256，随后一次性计算 MAE、RMSE 与 MaxAE。

Reference 三个 Area 为 899.934、935.736、450.478 kgCO2/MWh；强 Direct-M 的 MAE 为 140.160 kgCO2/MWh，与论文冻结事实一致。

## 4. v3.7 正式 SOTA 结果

| 方法 | 正式运行数 | MAE/(kgCO2/MWh) | RMSE | MaxAE | 相对 Direct-M MAE 降幅 |
|---|---:|---:|---:|---:|---:|
| Direct-M | 1 | 140.160 | 161.604 | 210.988 | — |
| SF* adapted | 1 | 0.769 | 0.853 | 1.061 | 99.45% |
| PM* adapted | 1 | 0.649 | 0.705 | 0.844 | 99.54% |
| PM-DSE adapted | 1 | 1.170 | 1.302 | 1.923 | 99.17% |
| PI-GNN adapted | 10 | 0.990 ± 0.186 | 1.124 ± 0.196 | 1.592 ± 0.364 | 99.29% |
| Full-ST（现有冻结结果） | 1 | 1.117 | 1.218 | 1.803 | 99.20% |

PI-GNN 10 个种子的 MAE 范围为 0.807—1.300 kgCO2/MWh。

## 5. 物理与执行审计

13 次正式预测全部满足公共物理投影约束。跨全部运行的最大区域月度用电、Area×源类型发电、分方向交换误差分别约为 0.2000000003、0.2000000000 和 0.2000000004 MWh，最大线路利用率为 1.0000000000000009；这些数值对应冻结的 0.2 MWh float32 数值带与求解器容差。

平均“全量训练+正式推理+公共投影”墙钟时间约为：SF* 5.47 s、PM* 7.43 s、PM-DSE 4.67 s、PI-GNN 17.67 s/seed。该时间只用于本工程环境内相对复现，不跨硬件解释。

## 6. 对论文结论的影响

这组结果必须原样保留，不能因为外部适配方法在 February 上更低而重新调参或更改特征。正式 February 结果显示，PM*、SF* 以及 PI-GNN 的 10-seed 平均 MAE 均低于当前 Full-ST；PM-DSE 略高于 Full-ST。因此论文不能写“Full-ST 全面优于近年 SOTA”或类似表述。

同时，比较的训练条件存在重要结构差异：四个学习/伪量测型适配基线使用 120 个目标期外、由同一物理后端产生的有标签 hidden-state 校准场景；Full-ST 不依赖这些有标签状态，也没有训练过程，仅使用目标月粗统计、固定物理边界和独立历史先验完成稀疏 LP 投影。因而本组实验更适合支撑“训练自由的物理方法在统一粗观测边界下达到与监督式状态构造方法同一误差量级”的结论，而不是以单一 February MAE 宣称精度领先。

正式结果已经通过 `V37_FORMAL_FREEZE.json` 冻结。后续不允许针对 February 修改比较器超参数、特征、seed、校准场景或公共投影。如果需要进一步判断跨时段泛化，应使用预先定义且不据 February 调整的新目标窗口进行扩展验证。
