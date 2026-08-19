from omnitransfer.unified_alignment import (
    TRANSFER_CONTRACT_SCHEMA_ID,
    TRANSFER_PIPELINE_MODULES,
    TransferRequest,
    transfer_contract_metadata,
)


def test_transfer_contract_describes_one_four_stage_pipeline() -> None:
    request = TransferRequest(target_xml="<hierarchy />", action_type="click", top_k=5)
    metadata = request.as_metadata()

    assert metadata["schema_id"] == TRANSFER_CONTRACT_SCHEMA_ID
    assert tuple(metadata["pipeline_modules"]) == TRANSFER_PIPELINE_MODULES
    assert transfer_contract_metadata()["coordinate_policy"] == (
        "relative_within_source_node_only"
    )
