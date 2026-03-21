from app.services.knowledge_service import knowledge_service


def test_classify_gov_doc_as_a_tier() -> None:
    classified = knowledge_service.classify(
        {
            "title": "\u56fd\u5bb6\u536b\u5065\u59d4\u53d1\u5e03\u5065\u5eb7\u79d1\u666e\u6587\u7ae0",
            "url": "https://www.nhc.gov.cn/test/article.html",
            "body_text": (
                "\u56fd\u5bb6\u536b\u5065\u59d4\u63d0\u793a\u516c\u4f17\u51fa\u73b0\u80f8\u75db\u5e94\u53ca\u65f6\u5c31\u533b\uff0c"
                "\u5fc5\u8981\u65f6\u524d\u5f80\u533b\u9662\u95e8\u8bca\u590d\u67e5\uff0c\u4e0d\u8981\u5ef6\u8bef\u6cbb\u7597\u3002"
            ),
        }
    )
    assert classified is not None
    assert classified.trust_tier == "A"


def test_reject_advertorial_page() -> None:
    classified = knowledge_service.classify(
        {
            "title": "\u67d0\u533b\u9662\u63a8\u5e7f\u5e7f\u544a",
            "url": "https://example.com/ad.html",
            "body_text": (
                "\u8fd9\u662f\u62db\u5546\u63a8\u5e7f\u5e7f\u544a\u9875\u9762\uff0c\u6b22\u8fce\u9884\u7ea6\u6302\u53f7\uff0c"
                "\u4e13\u5bb6\u5728\u7ebf\u95ee\u7b54\u3002"
            ),
        }
    )
    assert classified is None
