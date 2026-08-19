# Unified-error 长期合同

`unified-error` 是 OmniTransfer 的长期回归类别。它记录“模型看似有
视觉/布局证据，但局部语义关系被错误排序”的错误，必须持续通过，不能
在下一次重建 review 时被当成普通 pseudo-label 覆盖。

## 永久反例

Pinterest iOS profile 页面中，搜索框右侧有两个相邻控制：

```text
Display Options  ->  Android Sort boards by
Add              ->  Android Create
```

`Display Options` 不能映射到页面中下方的 `Find ideas`。这个反例专门
验证全页面绝对位置失效时，局部关系仍然有效。

## 证据优先级

从强到弱：

1. 可学习的局部语义与 affordance 证据；不得把某个 app 的词表写成
   固定角色规则。
2. 最近分叉父节点下的兄弟角色序列。
3. 直接子节点顺序及父节点内的局部几何。
4. 图像/图标视觉相似度，作为辅助证据。
5. 全页面绝对坐标和跨设备整体位置，只能作为弱 tie-breaker，不能主导结果。

规则实现的唯一 seam 是
`src/omnitransfer/unified_alignment.py`。PyTorch 训练前向和 NumPy 线上
推理共享同一个 strategy schema，策略权重从 correspondence labels 学习，
因此训练、运行时和 review 不得各自添加私有修正。

当前主路径不再包含等价页面捷径或 Spatial-XML 确定性改排；局部关系和
几何只作为统一 matcher 的可学习输入，避免模型输出之后再被旁路交换。

## 必过检查

```bash
PYTHONPATH=src /Users/wuzewen/Projects/Omni/OmniFlow-exp/.venv/bin/python \
  -m pytest tests/test_human_alignment_rules.py \
  tests/test_learned_matcher.py tests/test_numpy_v9_matcher.py -q
```

完整 matcher 修改还必须执行 `git diff --check`，并重新验证统一 review
中的实时点击结果。

## 每次训练后的错例审计

任何新 checkpoint 都不能只报告 Top-1。必须用
`scripts/build_model_error_review.py` 对逐节点预测做分层诊断，并把代表性
错例合并进已有 unified-error 队列。至少要分开统计：

- text/content-desc 两侧都有、只有一侧有、两侧都没有；
- encoder 已选对但 relation refinement 改错；
- 单侧文本、无文本局部语义、兄弟难负例和不同分支错误；
- gold rank 2–3、4–5 和大于 5；
- 模型文本证据明显强于 gold、需要人工复核标注的情况。

所有生成的 review 必须复用 `mapping_pair_review`，保留左右截图、G/M 点、
Source/Target 点击标注、点击 Source 后实时重算，以及节点内部相对点击位置
投影。不得生成只有表格或静态框的平行 review。

```bash
.venv/bin/python scripts/build_model_error_review.py \
  --dataset runtime/datasets/ase_human_gold_all_nodes_within_app_v1/test.jsonl \
  --predictions output/<run>/test.predictions.jsonl \
  --train runtime/datasets/ase_human_gold_all_nodes_within_app_v1/train.jsonl \
  --existing-review-dir output/unified_error_review \
  --cleaning-quarantine output/cleaned_train/quarantine.jsonl \
  --output output/unified_error_review_current_model
```

训练数据先通过 `scripts/clean_mapping_dataset.py`。清洗器不得使用
clickability，也不得扩展父子等价节点；可疑标签只进入 quarantine 和同一个
unified review，未经人工确认不得进入训练。
