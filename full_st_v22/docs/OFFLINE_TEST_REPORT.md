# 离线复现与 QA 报告（工程v2.2 / 论文v8）

QA日期：2026-09-09。

## 1 原始输入检查

`python scripts/validate_bundled_raw.py`：PASS。

- bus = 73 rows；
- branch = 120 rows；
- gen = 158 rows；
- regional load = 1368 h（Jan 1–14 + Feb 1–29 + Jun 1–14）；
- wind = 336 h（Jan 1–14）；
- 时间戳唯一，每个纳入日期均完整包含Period 1..24。

## 2 单元与端到端测试

`PYTHONPATH=src python -m pytest -q`：**9 passed**。

覆盖网络、CEF、隐状态LP、端到端尺度残差校正、捆绑原始数据、跨时段reference资产和有损AC审计。

## 3 本轮实际重跑

本轮实际重新执行并冻结：

- Jan-target / Jun-target 当前跨时段验证；
- Jan-wind 168 h压力场景；
- RTS-GMLC 73节点有损AC空间恒等式审计；
- `verify` 对自然月、风电、跨期与AC全部检查，`passed=true`。

696 h自然月代码和数据口径本版未改变；当前沙盒单次命令重跑受到执行时限影响，但v7已生成的同代码链自然月输出与当前reference在本轮`verify`中逐项通过。工程的一键入口已修复为顺序执行自然月→跨期→风电→AC→verify。

## 4 当前冻结结果

### 4.1 696 h自然月
- Direct-M MAE = 140.1596；
- Full-ST MAE = 1.1171 kgCO2/MWh；
- 降低 = 99.20%。

### 4.2 跨时段：固定Jan 1–7历史先验
- Jan-target：133.967 → 0.487 kgCO2/MWh，降低99.64%；
- Jun-target：61.150 → 6.213 kgCO2/MWh，降低89.84%。

### 4.3 Jan-wind
- Static-S / Load-T / Gen-T / Full-ST = 32.102 / 30.098 / 6.300 / 3.786 kgCO2/MWh；
- Full-ST残差相关系数 = 0.959。

### 4.4 有损AC审计
- 最大节点功率平衡残差 = 1.14e-13 MW；
- 总有功损耗 = 134.461 MW；
- 空间恒等式最大闭合误差 = 1.66e-10；
- 边界—网损分解最大闭合误差 = 1.66e-10；
- 边界暴露占比 = 96.54%。

## 5 版本边界

当前数值事实源：`reference/expected_*.csv`。旧Jan/Jun与旧case39/case118结果只存放于`reference/legacy_*`，用于证明历史实验资产存在，不进入论文v8的当前数值结论和自动核验。
