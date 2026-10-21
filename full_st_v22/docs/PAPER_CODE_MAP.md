# 论文—代码映射（v8实验资产恢复版）

| 论文内容 | 代码位置 |
|---|---|
| 节点有向潮流/网络矩阵 | `network.py` |
| 节点 CEF 方程 | `carbon.py::nodal_cef` |
| 细粒度月度 Reference | `aggregate.py::aggregate_state` |
| 小时区域重求解 H_m | `aggregate.py::aggregate_state` |
| 月度 Direct-M | `aggregate.py::area_direct_cef` |
| Δ^S、Δ^T、Δ | `aggregate.py::ScaleOutputs` |
| 历史星期类型—小时映射 | `priors.py::weekday_hour_map` |
| Static-S / Load-T / Gen-T / Full-ST | `experiments.py::METHODS` |
| 物理可行隐状态 LP | `hidden_state.py::project_generation` |
| Δ_hat 与月度反馈校正 | `experiments.py::run_natural_month` |
| 696 h自然月主实验 | `experiments.py::run_natural_month` |
| Jan/Jun跨时段验证 | `experiments.py::run_cross_period` |
| Jan-wind压力场景 | `experiments.py::run_wind_stress` |
| RTS-GMLC 73节点有损AC恒等式审计 | `ac_audit.py::audit_rts_ac` |
| 当前结果自动核验 | `verify.py::verify_outputs` |

论文中的“尺度残差等价类”属于任务定义，不需要在程序中显式实例化。代码在同一估计状态上分别计算细、粗责任路径，并以二者之差得到 `estimated_residual`，对应论文的尺度残差估计量。

`reference/expected_*.csv` 是当前v8唯一数值事实源；`reference/legacy_*` 只保留历史实验资产溯源，禁止与正文当前结果混用。
