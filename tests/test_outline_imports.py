def test_outline_imports() -> None:
    import omnitransfer.eval
    import omnitransfer.features
    import omnitransfer.importers
    import omnitransfer.matchers
    import omnitransfer.reports

    assert omnitransfer.reports.format_percent(0.767008) == "76.70%"
