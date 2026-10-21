# 固定区域月度电碳责任尺度残差校正 — 完整复现工程 v2.2 / 论文v8

对应论文《固定区域月度电碳责任尺度残差机理及校正》实验资产恢复版。

> 尺度非交换机理 → 物理可行隐状态 → 尺度残差估计 → 月度反馈校正

## 工程包含

- RTS-GMLC release 0.2官方原始源数据冻结子集；
- 2020年2月696 h完整自然月主实验；
- Jan/Jun两组168 h当前可复现跨时段验证；
- Jan-wind 168 h压力场景；
- 自包含RTS-GMLC 73节点有损AC恒等式审计；
- legacy Jan/Jun与case39/case118历史资产（仅溯源）；
- reference自动核验、单元测试、物理审计和论文文件。

## 安装与一键复现

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
python scripts/validate_bundled_raw.py
python -m pytest -q
python run_reproduce.py
```

标准复现离线运行。详细信息见 `docs/DATA_PROVENANCE.md`、`docs/REPRODUCIBILITY.md`、`docs/AC_AUDIT.md` 和 `docs/VERSION_NOTE_v8.md`。

## 当前论文事实源

1. `reference/expected_*.csv`：最高数值事实源；
2. `outputs/paper_main/`：本工程实际运行输出；
3. `reference/legacy_*`：历史溯源材料，禁止与当前正文混用。

### 核心结果

- Feb-2020 696 h：Direct-M `140.160` → Full-ST `1.117 kgCO2/MWh`（−99.20%）；
- Jan-target：`133.967` → `0.487`（−99.64%）；
- Jun-target：`61.150` → `6.213`（−89.84%）；
- Jan-wind Full-ST：`3.786 kgCO2/MWh`；
- 当前有损AC恒等式闭合误差约`1.66e-10`，边界暴露占比`96.54%`。

## 科学边界

隐状态用于生成与尺度残差有关的物理代表状态，不声称唯一恢复真实节点—小时状态。目标是提高区域月度消费侧电碳责任核算精度。AC审计断面为理论数值审计，不解释为实际运行排放场景。当前论文仍基于标准测试系统，尚未声称完成真实冀北实网外部验证。
