def test_package_exports_candidate_ranking_and_compatibility_transfer() -> None:
    import omnitransfer
    import omnitransfer.runtime as runtime

    assert omnitransfer.__all__ == ["action_transfer", "rank_action_candidates"]
    assert callable(omnitransfer.action_transfer)
    assert callable(omnitransfer.rank_action_candidates)
    assert not hasattr(omnitransfer, "describe_action_target")
    assert not hasattr(runtime, "describe_action_target")
    assert not hasattr(omnitransfer, "runtime_preflight")
    assert not hasattr(runtime, "runtime_preflight")
    assert not hasattr(runtime, "runtime_matcher_manifest")
