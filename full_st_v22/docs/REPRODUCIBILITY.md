# 复现说明（v8实验资产恢复版）

## 1 离线输入

本包包含当前全部实验所需RTS-GMLC release-0.2源数据冻结子集。先运行：

```bash
python scripts/validate_bundled_raw.py
```

## 2 一键复现

```bash
PYTHONPATH=src python run_reproduce.py
```

顺序：696 h自然月 → Jan/Jun跨时段验证 → Jan-wind → 自包含有损AC审计 → 与当前reference自动核验。

分步命令：

```bash
PYTHONPATH=src python -m scale_residual.cli natural-month --config configs/paper_main.toml
PYTHONPATH=src python -m scale_residual.cli cross-period --config configs/paper_main.toml
PYTHONPATH=src python -m scale_residual.cli wind-stress --config configs/paper_main.toml
PYTHONPATH=src python -m scale_residual.cli ac-audit --config configs/paper_main.toml
PYTHONPATH=src python -m scale_residual.cli verify --config configs/paper_main.toml
```

## 3 当前冻结结果

### 696 h自然月
Direct-M MAE≈140.160，Full-ST≈1.117 kgCO2/MWh，降低99.20%。

### 跨时段：同一Jan 1–7历史先验
- Jan-target (Jan 8–14): Direct-M `133.967`, Full-ST `0.487`, 降低`99.64%`；
- Jun-target (Jun 8–14): Direct-M `61.150`, Full-ST `6.213`, 降低`89.84%`。

June精度下降被保留为真实跨季节迁移结果，不通过目标期相关先验进行后验优化。

### Jan-wind
Static-S / Load-T / Gen-T / Full-ST MAE = `32.102 / 30.098 / 6.300 / 3.786 kgCO2/MWh`。

### 有损AC审计
空间恒等式与边界—网损分解最大闭合误差均约`1.66e-10`，边界暴露占比`96.54%`。

## 4 数值求解器与复杂度

DC-OPF与隐状态投影均由SciPy HiGHS求解；CEF为逐时线性方程。隐状态问题为标准稀疏LP，不含学习率、经验权重或自定义迭代。不同HiGHS版本可能在退化LP中选择不同等价最优基，因此核验针对区域月度责任与残差指标设置容差。

## 5 历史资产与当前事实源

- `reference/expected_*.csv`：当前论文v8唯一数值事实源；
- `reference/legacy_cross_period/`：已恢复旧Jan/Jun结果，仅作溯源；
- `reference/legacy_ac/`：旧case39/case118实验协议和结果摘要，仅作溯源。

不得将legacy数值重新混入当前正文或自动verify。
