---
exp_id: OT-G-260822-001
date: 2026-08-22
system: OmniTransfer
exp_type: ASE node correspondence disambiguation
dataset: ase2023_vision_based_widget_mapping
train_page_pairs: 839
dev_page_pairs: 115
dev_supervised_rows: 1566
device: NVIDIA GeForce RTX 4090
seed: 17
status: complete
tags: [ASE, local-correspondence, hard-negative, visual, exact-order]
---

# 实验目的

验证 Top-1/Top-2 局部消歧的三个具体改动是否有效，同时保持一条端到端路径：

1. 精确局部顺序与四头局部对应；
2. 最终 Rank-1/Rank-2 行内困难样本加权；
3. 稳定固定视觉描述之上的轻量可学习紧裁剪与上下文残差。

# 不变合同

- 仅使用 ASE 人工节点 correspondence gold；resource-id 和 node-id 只取标签，不进模型。
- 一个对称节点 correspondence cross-entropy；无 NULL、bonus、rerank 或额外推理 loss。
- 三层 correspondence-conditioned refinement；每页最多保留 5 个学习选择的邻居。
- 使用同一个 v9 multimodal/contextual checkpoint 初始化；所有参数联合训练。

# 观察

- Dev 的 230/230 张页面均有截图。
- 1,676/1,676 个检查到的受监督 source/target endpoint 均有 bbox。
- 固定视觉模型对 gold 无文字 endpoint 的平均视觉路由权重为 60.45%，中位数为 61.82%；视觉证据已被大量使用，但此前不可学习。
- 旧硬邻居候选仅让 558/729（76.54%）个 source 监督行看到至少一个双边 gold 局部支持，222/729（30.45%）看到至少三个；这是下界，因为 ASE 只标注部分页面节点对应。

# 结果

| 实验 | Dev Top-1 | Dev Top-2 | Rank-2 | 结论 |
|---|---:|---:|---:|---|
| 新 clean 基线 | 78.22% (1225/1566) | 86.27% (1351/1566) | 126 | 对照 |
| 旧最佳模型 | 78.67% (1232/1566) | 86.33% (1352/1566) | 120 | 历史对照 |
| 精确顺序 + 4 头局部对应 + hard-row=3 | **78.80% (1234/1566)** | 86.21% (1350/1566) | **116** | Top-1 增加 9 条，但 Top-2 少 1 条；局部最终交换改善，召回未改善 |
| 可学习视觉残差，误用 0.1× LR | 78.48% (1229/1566) | 86.46% (1354/1566) | 125 | 新视觉被误归入旧 v9 低学习率组；失败对照 |
| 可学习视觉残差，完整 LR，旧 0.25 候选 | **79.05% (1238/1566)** | **86.78% (1359/1566)** | **121** | 比 clean 基线多 13 条 Top-1；视觉有效但仍未解决大部分最后交换 |
| 全页候选 + learned Top-5 | **79.76% (1249/1566)** | **87.74% (1374/1566)** | 125 | 比同配置 0.25 候选多 11 条 Top-1、15 条 Top-2；连续加权优于硬资格门槛 |

发布评估器对最终 checkpoint 的复扫为 Top-1 **79.63%**、Top-3 **90.87%**、Top-5 **94.25%**。训练器固定验证复扫与发布评估器相差 2 个 Top-1 行；在二者口径统一前，对外应采用更保守的 79.63%。

# 异常与失败经验

- 初始 checkpoint 选择曾错误使用“Top-1 后 Rank-2 更少”作为 tie-break。Rank-2 减少也可能意味着 gold 掉到 Rank-3+，因此已修正为 Top-1、Top-2、loss 的顺序。
- epoch 内 Train 指标是在线 pre-update aggregate，不能和固定 checkpoint 重扫混用。报告已明确区分两者。
- 固定视觉能缩小候选，但不能从 ASE gold 学习 icon、wrapper 与上下文；不能据此判断视觉已充分利用。
- 新视觉残差和局部顺序层最初被参数名前缀误归为“已迁移 v9 参数”，只得到 `3e-5` 学习率；修正后新视觉分支使用 `3e-4`，真正迁移的 v9 层仍使用 `3e-5`。
- 原实现对候选池使用 `structure OR distance<=0.25` 硬门槛。可学习权重只发生在门槛内部，不符合“谁是好邻居也要学习”的要求。
- 训练器和发布评估器对同一最终 checkpoint 的 Top-1 相差 2 行（1249 对 1247）。这不改变 A/B 方向，但属于指标合同问题，正式验收前必须逐行对齐。

# 下一步与判据

1. 完成固定视觉与可学习视觉的同 seed、同 loss、同 epoch 消融。
2. 下一步只验证 candidate-conditioned neighbour weighting：针对正在比较的候选对选择 witness，而不是先做 page-only Top-5。不要再扩 CNN 或增加规则分数。
3. 单独报告 textless、near-confusion 与页面缓存后的 `encode_page` / `match_page` 延迟。

# 原始材料

- `output/rank2_disambiguation_20260822_r001/full8/report.json`
- `output/rank2_disambiguation_20260822_r001/full8/history.jsonl`
- `output/rank2_disambiguation_20260822_r001/visual_full8_full_lr_hardcut_r002/report.json`
- `output/rank2_disambiguation_20260822_r001/visual_full8_full_lr_all_candidates/report.json`
- 4090：`/home/zewen/omnitransfer_runs/rank2_disambiguation_20260822_r001/output/`
