# 工程化设计（v2.2）

核心层级：

1. `data.py`：捆绑原始数据、数据路径与日期切片；
2. `generator_profile.py`：论文冻结机组兼容配置；
3. `network.py`：DC网络、PTDF、节点负荷下分；
4. `dispatch.py`：Reference/历史状态DC-OPF；
5. `carbon.py`：节点CEF；
6. `aggregate.py`：细路径、小时区域路径、月度Direct路径与空间/时间/总尺度残差；
7. `priors.py`：独立历史先验；
8. `hidden_state.py`：物理可行稀疏LP投影；
9. `experiments.py`：696 h自然月、Jan/Jun跨期、Jan-wind实验编排；
10. `ac_audit.py`：自包含RTS-GMLC 73节点有损AC空间恒等式审计；
11. `verify.py`：自然月、风电、跨期和AC结果与当前reference自动比对；
12. `reference/legacy_*`：历史实验资产，仅作来源追溯。

研究代码与原始数据、配置、输出、当前reference和历史资产分离，避免 notebook 式隐式状态和版本数字混用。
