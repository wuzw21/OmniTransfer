# OmniTransfer 代码清理结果

日期：2026-08-17
状态：核心代码、脚本、历史模型、兼容接口和重复审核器已经收敛；未创建 Git commit。

## 1. 现在只保留什么

OmniTransfer 只负责一件事：

```text
source XML + source node/point + target XML
    -> 完整排序的 target candidates + evidence
```

唯一公开运行时接口：

```python
from omnitransfer import rank_action_candidates
```

唯一运行链：

```text
rank_action_candidates
  -> runtime.py
  -> NumpyGeometricAlignmentMatcher
  -> 固定 direct-text geometric-v9 NPZ
```

唯一离线链：

```text
omnitransfer.ui_correspondence_pair.v1
  -> build_ui_correspondence_dataset.py
  -> review_annotation_template.html
  -> train_geometric_v9_matcher.py
  -> evaluate_geometric_v9_matcher.py
  -> export_runtime_checkpoint.py
```

UTG 数据只保留一个到 canonical review contract 的转换入口：

- `scripts/build_utg_point_mapping_review.py`
- `scripts/utg_mapping_candidates.py`（私有 helper）

它们不采集设备、不启动 DroidBot，也不实现第二套 HTML。

## 2. 已删除内容

### 历史模型与实现

- mutual v2/v3 matcher、NumPy mutual matcher 和 64D anchor baseline。
- legacy relation-bias、LightGlue、v4–v8、v10–v18、fusion、transformer、consensus 等历史分支。
- context/local-alignment/geometry-refinement 的独立旧架构入口。
- 历史 checkpoint 和对应测试。
- 旧 Query 模型、fine-tune、benchmark、disagreement、outcome-learning 路径。

当前 PyTorch 模型直接构建 geometric-v9，不再先伪装成旧 context 架构后追加 reranker。

### 兼容接口

- `action_transfer`。
- `OmniTransferMatcher`、`LearnedGraphMatcher`。
- `build_omnitransfer_matcher`、`build_relation_aware_matcher`。
- `RelationAwareMatcher`。
- actionable candidate policy。
- legacy feature schema 自动分派与旧 schema 升级。
- 旧 warm-start、跨架构 checkpoint 迁移和 architecture CLI。

当前内部名称统一为 geometric-v9：

- `build_geometric_v9_matcher`
- `GeometricMatcher`
- `train_geometric_v9_matcher`

### 重复数据与审核路径

- AndroidControl、GUIOdyssey、MobileViews、OpenMobile、OS-Atlas、WebUI、Mind2Web、unified-UI 等 source-specific importer。
- widget、GUIOdyssey、cluster、pilot 等独立 reviewer 和 HTML 生成器。
- 历史 dataset registry、Query schema、compact bundle 与重复 split/eval 工具。

所有审核页面现在必须使用：

```text
tests/vector/review_annotation_template.html
```

### 设备采集与实验脚本

- Android/UTG/DroidBot 探索器、批处理器、设备 setup 和采集测试。
- 一次性 pilot、grid、ablation、external baseline、download helper。
- 旧 train/evaluate/export 入口。

OmniTransfer 仓库不再拥有 emulator 生命周期或 Android 采集职责。

### 无效训练选项

删除了当前 geometric-v9 不使用或默认关闭的历史 objective：

- teacher distillation
- cycle loss
- collision loss
- hard-negative margin loss
- edge-relation loss
- descriptor warmup
- context masking
- semantic dropout

当前训练 objective 只有完整 all-node assignment 与 node-descriptor loss。

## 3. 保留的 checkpoint

三个文件各有不同职责，不是重复副本：

1. `v9_direct_text_alignment_seed29.npz`：正式 NumPy runtime。
2. `v9_direct_text_alignment_seed29.pt`：runtime 导出来源和 Torch/NumPy 等价验证。
3. `v9_spatial_xml_alignment_seed29.pt`：OmniFlow page embedding。

运行时不会扫描目录、选择实验模型或回退到旧 checkpoint。

## 4. 当前模块

```text
src/omnitransfer/
  __init__.py                 唯一公开接口
  runtime.py                  候选生成边界
  numpy_v9_matcher.py         固定 NumPy runtime
  learned_matcher.py          特征、checkpoint、Torch adapter
  geometric_matcher.py        唯一 trainable model
  page_embedding.py           geometric-v9 page embedding
  ui_graph.py                 UI graph contract
  mapping_dataset.py          canonical correspondence contract/split
  mapping_pair_review.py      唯一 review renderer
  mapping_training.py         canonical dataset -> training pair
  self_supervised.py          augmentation、objective、trainer、evaluator
  experiment_logging.py       训练/评测 provenance
```

安装基础包只需要 NumPy 和 Pillow；PyTorch 已移入 `train` optional dependency，避免纯 runtime 被训练环境绑定。

## 5. 明确保留但不清理的数据

以下属于用户数据或运行产物，没有删除：

- `output/`
- `tmp/`
- review annotations
- screenshots、XML、JSONL
- 用户生成的数据集和审核结果

`.playwright-cli/`、`.DS_Store` 等本地文件未作为源代码处理。

## 6. 验证

执行：

```bash
python -m py_compile src/omnitransfer/*.py scripts/*.py
PYTHONPATH=src:. pytest -q
```

结果：58 个测试通过；4 个 Torch 相关测试因本机已有的
`libtorch_cpu.dylib` 缺失而跳过。该环境问题不是本次清理引入，未在代码清理中修改本机 PyTorch。

## 7. 不再允许重新引入

- 第二个公开 runtime operation。
- source-coordinate passthrough 或隐藏 fallback。
- 第二套 dataset schema、review template、trainer、evaluator、release exporter。
- architecture switch、历史模型兼容 alias、source-specific importer。
- emulator/UTG/DroidBot 采集实现。
- 在主 package 中恢复论文 pilot、ablation 或历史 benchmark。
