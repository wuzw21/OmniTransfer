from omnitransfer.unified_alignment import (
    ALIGNMENT_STRATEGY_SCHEMA_ID,
    default_alignment_strategy_registry,
)
from omnitransfer.ui_graph import graph_from_record


SOURCE_XML = """
<hierarchy bounds="[0,0][1,1]" coordinate-space="xml-relative-0-1">
  <node node-id="0.0" class="XCUIElementTypeApplication" bounds="[0,0][1,1]">
    <node node-id="header" class="XCUIElementTypeOther" bounds="[0,.10][1,.18]">
      <node node-id="search" text="Search your Pins" class="XCUIElementTypeOther" bounds="[.02,.11][.76,.16]" />
      <node node-id="display" text="Display Options" class="XCUIElementTypeButton" bounds="[.80,.11][.86,.16]" />
      <node node-id="add" text="Add" class="XCUIElementTypeButton" bounds="[.88,.11][.94,.16]" />
    </node>
  </node>
</hierarchy>
"""

TARGET_XML = """
<hierarchy bounds="[0,0][1,1]" coordinate-space="xml-relative-0-1">
  <node node-id="0.0" class="android.widget.FrameLayout" bounds="[0,0][1,1]">
    <node node-id="controls" class="android.widget.LinearLayout" bounds="[0,.42][1,.50]">
      <node node-id="search" text="Search your Pins" class="android.widget.TextView" clickable="true" bounds="[.02,.43][.74,.49]" />
      <node node-id="filter" content-desc="Sort boards by" resource-id="com.pinterest:id/filter" class="android.widget.ImageView" clickable="true" bounds="[.76,.43][.87,.49]" />
      <node node-id="create" content-desc="Create" resource-id="com.pinterest:id/create" class="android.widget.ImageView" clickable="true" bounds="[.88,.43][.98,.49]" />
    </node>
    <node node-id="empty" class="android.widget.FrameLayout" bounds="[0,.55][1,.75]">
      <node node-id="find" text="Find ideas" class="android.widget.Button" clickable="true" bounds="[.35,.60][.65,.66]" />
    </node>
  </node>
</hierarchy>
"""


def test_unified_error_uses_generic_learnable_strategy_contract() -> None:
    """The Pinterest regression must not be represented by an app word list."""

    source = graph_from_record({"xml": SOURCE_XML}, graph_id="source")
    target = graph_from_record({"xml": TARGET_XML}, graph_id="target")
    registry = default_alignment_strategy_registry(
        direct_pair_dimension=18,
        typed_relation_dimension=16 * 16,
        geometry_dimension=16,
    )

    assert registry.schema_id == ALIGNMENT_STRATEGY_SCHEMA_ID
    assert registry.dimension == 18 + 16 + 16 * 16 + 16
    assert all(
        forbidden not in " ".join(registry.names).lower()
        for forbidden in ("pinterest", "display", "filter", "create", "find ideas")
    )
    assert source.nodes and target.nodes
