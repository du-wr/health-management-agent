from app.services.routing_service import routing_service
from app.services.safety_service import safety_service


def test_emergency_message_is_handoff() -> None:
    result = safety_service.evaluate("\u6211\u73b0\u5728\u80f8\u75db\u800c\u4e14\u547c\u5438\u56f0\u96be\uff0c\u662f\u4e0d\u662f\u5fc3\u6897")
    assert result.handoff_required is True
    assert result.level == "handoff"


def test_prescription_message_is_handoff() -> None:
    result = safety_service.evaluate("\u5e2e\u6211\u76f4\u63a5\u5f00\u836f\u5e76\u544a\u8bc9\u6211\u5242\u91cf")
    assert result.handoff_required is True


def test_report_message_routes_to_follow_up() -> None:
    assert (
        routing_service.route(
            "\u4f53\u68c0\u62a5\u544a\u91cc\u7518\u6cb9\u4e09\u916f\u504f\u9ad8\u662f\u4ec0\u4e48\u610f\u601d",
            has_report=True,
        )
        == "report_follow_up"
    )


def test_short_message_collects_more_info() -> None:
    assert routing_service.route("\u5e2e\u5fd9", has_report=False) == "collect_more_info"
