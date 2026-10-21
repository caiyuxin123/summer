# 冀北双链论文 SOTA 状态构造比较器 v3.7

v3.7 是论文一致型 SOTA 正式 February 盲测冻结包。四个主比较器为 PI-GNN adapted、SF* adapted、PM* adapted、PM-DSE adapted；均先输出 168 h×48 候选细粒度状态，再进入同一月度统计/DC物理投影、节点 CEF 和固定 Area 聚合。

正式评价按两阶段执行：Phase A 只接入 `target_input_only.npz` 并冻结 13 组预测；Phase B 在 Phase-A manifest 固定后才一次性接入 `reference_only.npz`。正式结果位于 `formal_eval/FORMAL_FEB_SOTA_RESULTS_v3.7.json`，冻结关系位于 `formal_eval/V37_FORMAL_FREEZE.json`。

注意：v3.5 direct-y 结果仍只作历史诊断，不进入论文。v3.7 之后不得依据 February 结果重新调参或更改比较协议。
