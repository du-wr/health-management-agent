from __future__ import annotations

import json
from typing import Any


def report_ocr_prompt() -> str:
    return (
        "你是中文体检报告 OCR 助手。"
        "请按阅读顺序提取图片中的原始文字，尽量保留项目名、数值、单位、参考范围、超声结论和医生提示。"
        "不要解释，不要补充，只输出识别到的文本。"
    )


def lab_extraction_system_prompt() -> str:
    return (
        "你是中文体检报告结构化抽取助手。"
        "请从报告文本中识别检验或体检项目，并返回 JSON 对象，顶层字段只允许为 items。"
        "items 中每一项都必须包含 name, value_raw, value_num, unit, reference_range, clinical_note。"
        "如果 value_num 无法判断，则填写 null。"
    )


def intent_router_system_prompt() -> str:
    return (
        "你是医疗健康咨询 Agent 的意图分类器。"
        "请把用户问题分类到以下四类之一："
        "report_follow_up, term_explanation, symptom_rag_advice, collect_more_info。"
        "返回 JSON，对象字段为 intent, reason。"
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


def input_analysis_system_prompt() -> str:
    return (
        "你是医疗健康咨询 Agent 的输入分析器。"
        "你会收到用户问题、是否存在报告上下文以及最近几轮对话。"
        "请先理解用户真实意图，再把问题改写成更适合检索和后续回答的形式。"
        "返回 JSON，对象字段必须包含："
        "intent, rewritten_query, normalized_term, use_local_knowledge, use_who, use_max, reason。"
        "规则："
        "1. intent 只能是 report_follow_up, term_explanation, symptom_rag_advice, collect_more_info。"
        "2. rewritten_query 用于后续本地知识检索，应去掉口语噪声、代词和无意义尾句。"
        "3. normalized_term 仅在术语解释场景填写，尽量改写成标准医学名词。"
        "4. term_explanation 且问题涉及疾病、综合征或诊断名时，通常 use_who=true。"
        "5. 涉及报告解读、术语解释、症状建议时，通常 use_max=true；信息明显不足时可为 false。"
        "6. 只做路由和改写，不输出正式医学答案。"
    )


def input_analysis_user_prompt(message: str, has_report: bool, conversation_history: list[dict[str, Any]]) -> str:
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
        "请返回 JSON，对象顶层字段只允许为 items。"
        "items 中每一项都必须包含：name, meaning, common_reasons, watch_points, suggested_department。"
        "只做健康信息解释，不做诊断。解释要结合当前报告上下文，而不是只复述知识库。"
    )


def report_follow_up_planner_system_prompt() -> str:
    return (
        "你是体检报告追问拆解器。"
        "你会收到用户当前问题、最近几轮对话、报告中的重点异常项和相关指标。"
        "请先判断用户这轮真正想解决什么，再把回答任务拆成结构化计划。"
        "返回 JSON，对象字段必须包含："
        "focus_item_names, need_item_explanations, need_synthesis, need_next_steps, "
        "synthesis_axes, follow_up_needed, reason。"
        "规则："
        "1. focus_item_names 应优先选择与当前问题最相关的 1 到 6 个报告项目名。"
        "2. need_item_explanations 表示是否需要逐项解释这些指标异常代表什么。"
        "3. need_synthesis 表示是否需要解释这些异常合起来说明什么。"
        "4. need_next_steps 表示是否需要给出复查、生活方式或建议科室。"
        "5. synthesis_axes 用短语列出综合解读方向，例如“血脂代谢风险”“肾功能背景”“贫血方向线索”。"
        "6. 如果用户问题太模糊，可将 follow_up_needed 设为 true，但仍尽量利用现有报告给出初步解读。"
        "7. 不要输出正式医学答案，只输出计划。"
    )


def report_follow_up_planner_user_prompt(
    message: str,
    conversation_history: list[dict[str, Any]],
    focus_items: list[dict[str, Any]],
    related_items: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "message": message,
            "conversation_history": conversation_history,
            "focus_items": focus_items,
            "related_items": related_items,
        },
        ensure_ascii=False,
    )


def report_synthesis_system_prompt() -> str:
    return (
        "你是体检报告综合解读整理器。"
        "你会收到当前问题、报告追问拆解计划、逐项指标解释和相关指标。"
        "请把这些信息整理成结构化综合素材，而不是直接写最终答案。"
        "返回 JSON，对象字段必须包含：summary, priority_axes, combined_findings, next_steps。"
        "规则："
        "1. summary 用 1 到 2 句概括当前报告最值得优先关注的方向。"
        "2. priority_axes 是 1 到 4 个短语。"
        "3. combined_findings 用 2 到 4 条短句说明这些异常合起来提示什么，但不要下诊断。"
        "4. next_steps 用 2 到 4 条短句给出复查、生活方式或建议科室方向，不要给处方和剂量。"
        "5. 如果信息不足，请明确写出不确定性，但仍尽量做有限整理。"
    )


def report_synthesis_user_prompt(
    *,
    message: str,
    plan: dict[str, Any],
    interpretations: list[dict[str, Any]],
    related_items: list[dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "message": message,
            "plan": plan,
            "interpretations": interpretations,
            "related_items": related_items,
        },
        ensure_ascii=False,
    )


def report_answer_polish_system_prompt() -> str:
    return (
        "你是体检报告解读润色助手。"
        "你会收到一份已经结构完整的报告解读草稿。"
        "请在不改变医学边界的前提下，把它润色成更自然、清晰、易读的中文回答。"
        "要求："
        "1. 保留原有结构，不要删掉关键段落。"
        "2. 不要输出空标题，不要只写提纲。"
        "3. 可以优化衔接和表达，但不要新增诊断、处方、剂量。"
        "4. 如果草稿已经完整，只做轻量润色。"
    )


def report_answer_polish_user_prompt(
    *,
    message: str,
    draft_answer: str,
    plan: dict[str, Any],
    synthesis: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "message": message,
            "draft_answer": draft_answer,
            "plan": plan,
            "synthesis": synthesis,
        },
        ensure_ascii=False,
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
        "2. 如果当前问题依赖上下文，请结合最近几轮对话补全指代。"
        "3. 报告解读优先按“异常指标分别代表什么、这些异常合起来说明什么、接下来关注什么”组织。"
        "4. 如有超声或影像发现，也要单独说明。"
        "5. 可以给生活方式和复查方向建议，但不能下诊断、不能开药、不能给剂量。"
        "6. 输出清晰段落和列表，不要只写提纲标题。"
        "7. 如果工具结果不足以支持综合判断，就明确说明不确定性，不要编造。"
    )


def term_explanation_system_prompt() -> str:
    return (
        "你是医学名词解释助手。"
        "你会收到用户提问、本地知识库内容，以及可选的 WHO ICD-11 检索结果。"
        "请优先参考 WHO ICD-11 结果解释疾病或医学名词，再结合本地知识库，用普通人能理解的中文输出。"
        "固定按以下结构组织："
        "1. 它是什么 "
        "2. 常见表现或特点 "
        "3. 常见诱因或易感因素 "
        "4. 什么时候需要就医 "
        "5. 温馨提示。"
        "每一节都必须有实质内容，不能只写标题。"
        "不要输出处方、剂量或个体化治疗方案。"
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
        "请在不改变医疗边界的前提下，把半成品回答补全成一段完整、自然、可读的中文回答。"
        "不要输出空标题，不要保留只有冒号却没有内容的小节。"
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
        "1. 异常指标摘要 "
        "2. 综合解读与风险提示 "
        "3. 生活方式建议 "
        "4. 推荐就医科室。"
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
