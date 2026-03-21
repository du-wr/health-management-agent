from app.services.report_service import report_service


def test_determine_status_between_range() -> None:
    assert report_service._determine_status(6.2, "3.9-6.1") == "high"
    assert report_service._determine_status(3.2, "3.9-6.1") == "low"
    assert report_service._determine_status(4.8, "3.9-6.1") == "normal"


def test_determine_status_single_bound() -> None:
    assert report_service._determine_status(5.8, "< 5.7") == "high"
    assert report_service._determine_status(42.0, ">= 40") == "normal"
    assert report_service._determine_status(30.0, ">= 40") == "low"

