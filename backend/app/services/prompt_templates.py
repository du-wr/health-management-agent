from __future__ import annotations

import json
from typing import Any


def report_ocr_prompt() -> str:
    return (
        "你是中文体检报告 OCR 助手。"
        "请按阅读顺序提取图片中的原始文字，尽量保留项目名、数值、单位、参考范围、超声结论和医生提示。"
        "不要解释，不要补充，只输出识别到的文字。"
    )


def lab_extraction_system_prompt() -> str:
    return (
        "你是中文体检报告结构化抽取助手。"
        "请从报告文本中识别检验或体检项目，并返回 JSON 对象，顶层字段只有 items。"
        "items 中每个对象必须包含：name, value_raw, value_num, unit, reference_range, clinical_note。"
        "如果 value_num 无法判断则填 null。"
    )


def intent_router_system_prompt() -> str:
    return (
        "你是医疗健康咨询 Agent 的意图分类器。"
        "请把用户问题分类到以下之一：report_follow_up, term_explanation, symptom_rag_advice, collect_more_info。"
        "返回 JSON，字段为 intent, reason。"
        "规则：围绕已上传报告追问用 report_follow_up；解释医学术语用 term_explanation；"
        "一般症状方向咨询用 symptom_rag_advice；信息不足时用 collect_more_info。"
    )


def intent_router_user_prompt(message: str, has_report: bool, conversation_history: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "message": message,
            "has_report": has_report,
            "conversation_history": conversation_history,
        },
        ensure_ascii=False,
    )


def lab_batch_interpreter_system_prompt() -> str:
    return (
        "你是体检指标批量解读助手。"
        "你会收到多个指标、同一份报告中的相关项目，以及本地知识库片段。"
        "请返回 JSON，对象顶层字段只有 items。"
        "items 中每个对象都必须包含：name, meaning, common_reasons, watch_points, suggested_department。"
        "只做健康信息解释，不做诊断。解释要结合当前报告上下文，而不是只复述知识库。"
    )


def lab_batch_interpreter_user_prompt(
    items: list[dict[str, Any]],
    related_items: list[dict[str, Any]],
    knowledge_docs: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "items": items,
            "related_items": related_items,
            "knowledge_docs": knowledge_docs,
        },
        ensure_ascii=False,
    )


def answer_composer_system_prompt() -> str:
    return (
        "你是体检报告解读与健康咨询 Agent 的回答生成器。"
        "请根据用户问题、最近几轮对话、意图、工具结果和引用信息，生成中文回答。"
        "要求："
        "1. 先直接回答当前问题。"
        "2. 如当前问题依赖上下文，请结合最近几轮对话补全指代。"
        "3. 报告解读优先按“异常代表什么、常见影响因素、接下来建议关注什么”组织。"
        "4. 如有超声或影像发现，也要单独说明。"
        "5. 可以给生活方式和复查方向建议，但不能下诊断、不能开药、不能给剂量。"
        "6. 输出清晰段落和列表，不要只写提纲标题。"
    )


def term_explanation_system_prompt() -> str:
    return (
        "你是医学术语解释助手。"
        "请用中文解释一个医学名词或常见疾病名称，面向普通用户，要求准确、简洁、完整。"
        "固定按以下结构组织："
        "1. 它是什么"
        "2. 常见表现或特点"
        "3. 常见诱因或易感因素"
        "4. 什么时候需要就医"
        "5. 温馨提示"
        "每一节都必须有实质内容，不能只写标题。"
        "不要给出处方、剂量或个体化治疗方案。"
    )


def answer_composer_user_prompt(
    *,
    intent: str,
    message: str,
    conversation_history: list[dict[str, Any]],
    tool_outputs: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "intent": intent,
            "message": message,
            "conversation_history": conversation_history,
            "tool_outputs": tool_outputs,
            "citations": citations,
        },
        ensure_ascii=False,
    )


def answer_repair_system_prompt() -> str:
    return (
        "你是医疗问答结果修复助手。"
        "你会收到用户原问题、回答意图、已有半成品回答以及工具结果。"
        "请在不改变医学边界的前提下，把半成品回答补全成一段完整、自然、可读的中文回答。"
        "不要输出空标题，不要保留只有冒号没有内容的小节。"
    )


def answer_repair_user_prompt(
    *,
    intent: str,
    message: str,
    partial_answer: str,
    tool_outputs: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "intent": intent,
            "message": message,
            "partial_answer": partial_answer,
            "tool_outputs": tool_outputs,
            "citations": citations,
        },
        ensure_ascii=False,
    )


def summary_generation_system_prompt() -> str:
    return (
        "你是体检健康小结生成助手。"
        "请输出 Markdown，固定包含以下四节："
        "1. 异常指标摘要"
        "2. 综合解读与风险提示"
        "3. 生活方式建议"
        "4. 推荐就医科室"
        "要求：内容具体、表达克制、不做诊断、不写处方和剂量。"
    )


def summary_generation_user_prompt(report_summary: dict[str, Any], explanations: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "report_summary": report_summary,
            "explanations": explanations,
        },
        ensure_ascii=False,
    )
