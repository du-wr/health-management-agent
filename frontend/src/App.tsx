import { ChangeEvent, Fragment, FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";

import {
  createSession,
  deleteSession,
  generateSummaryForSession,
  getReport,
  getSessionDetail,
  getSessionMessages,
  getSessionSummaries,
  listSessions,
  renameSession,
  streamChat,
  streamReportProgress,
  uploadReportToSession,
} from "./api";
import type {
  AgentResponse,
  ChatStreamEvent,
  Citation,
  ReportParseResult,
  ReportProgressPayload,
  SessionMessage,
  SessionSummary,
  SummaryArtifact,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api";

type MessageRow = {
  id: string;
  role: "user" | "assistant";
  content: string;
  intent?: string | null;
  citations?: Citation[];
  meta?: AgentResponse;
  statusLabel?: string | null;
  streaming?: boolean;
  createdAt?: string;
};

const INTENT_LABELS: Record<string, string> = {
  report_follow_up: "报告追问",
  term_explanation: "术语解释",
  symptom_rag_advice: "症状建议",
  collect_more_info: "补充信息",
  safety_handoff: "安全转诊",
};

const SAFETY_LABELS: Record<string, string> = {
  safe: "安全",
  caution: "需谨慎",
  handoff: "建议线下就医",
};

// 这里只做轻量级加粗渲染，避免为了当前工作台额外引入 Markdown 运行时依赖。
function renderInline(text: string): ReactNode[] {
  const parts = text.split(/(\*\*.*?\*\*)/g).filter(Boolean);
  return parts.map((part, index) => {
    const match = /^\*\*(.*?)\*\*$/.exec(part);
    if (match) {
      return <strong key={`${part}-${index}`}>{match[1]}</strong>;
    }
    return <Fragment key={`${part}-${index}`}>{part}</Fragment>;
  });
}

function renderParagraph(text: string, className?: string): ReactNode {
  return <p className={className}>{renderInline(text)}</p>;
}

// 对助手消息做简单的结构化排版，支持标题、列表和提示块，保证医疗回答更易扫描。
function renderMessageContent(content: string): ReactNode {
  const normalized = content.replace(/\r\n/g, "\n").trim();
  if (!normalized) {
    return null;
  }

  const blocks = normalized.split(/\n\s*\n/);
  return blocks.map((block, blockIndex) => {
    const lines = block
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

    if (lines.length === 0) {
      return null;
    }

    const heading = /^(#{1,3})\s+(.+)$/.exec(lines[0]);
    if (heading) {
      const level = heading[1].length;
      const title = heading[2];
      if (level === 1) {
        return (
          <h3 className="message-title" key={`block-${blockIndex}`}>
            {renderInline(title)}
          </h3>
        );
      }
      if (level === 2) {
        return (
          <h4 className="message-subtitle" key={`block-${blockIndex}`}>
            {renderInline(title)}
          </h4>
        );
      }
      return (
        <h5 className="message-minor-title" key={`block-${blockIndex}`}>
          {renderInline(title)}
        </h5>
      );
    }

    const unorderedItems = lines
      .map((line) => /^[-*]\s+(.+)$/.exec(line))
      .filter((item): item is RegExpExecArray => item !== null);
    if (unorderedItems.length === lines.length) {
      return (
        <ul className="message-list" key={`block-${blockIndex}`}>
          {unorderedItems.map((item, index) => (
            <li key={`${item[1]}-${index}`}>{renderInline(item[1])}</li>
          ))}
        </ul>
      );
    }

    const orderedItems = lines
      .map((line) => /^\d+\.\s+(.+)$/.exec(line))
      .filter((item): item is RegExpExecArray => item !== null);
    if (orderedItems.length === lines.length) {
      return (
        <ol className="message-list ordered" key={`block-${blockIndex}`}>
          {orderedItems.map((item, index) => (
            <li key={`${item[1]}-${index}`}>{renderInline(item[1])}</li>
          ))}
        </ol>
      );
    }

    const merged = lines.join(" ");
    if (/^(重要提示|温馨提示|注意|说明)[:：]/.test(merged)) {
      return (
        <div className="message-note" key={`block-${blockIndex}`}>
          {renderParagraph(merged)}
        </div>
      );
    }

    return (
      <div className="message-paragraph-group" key={`block-${blockIndex}`}>
        {lines.map((line, index) => (
          <Fragment key={`${line}-${index}`}>{renderParagraph(line)}</Fragment>
        ))}
      </div>
    );
  });
}

function getStreamEventData<T>(event: ChatStreamEvent): T {
  return event.data as T;
}

function formatDebugValue(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function reportStatusText(status: string): string {
  if (status === "uploaded") return "已上传";
  if (status === "queued") return "排队中";
  if (status === "processing") return "解析中";
  if (status === "parsed") return "已完成";
  if (status === "needs_review") return "需复核";
  if (status === "error") return "失败";
  return status;
}

function reportTone(status?: string | null): "ready" | "working" | "warning" | "error" | "idle" {
  if (!status) return "idle";
  if (status === "parsed") return "ready";
  if (status === "needs_review") return "warning";
  if (status === "error") return "error";
  if (["uploaded", "queued", "processing"].includes(status)) return "working";
  return "idle";
}

function intentText(intent?: string | null): string {
  if (!intent) {
    return "未分类";
  }
  return INTENT_LABELS[intent] ?? intent;
}

function safetyText(level?: string | null): string {
  if (!level) {
    return "未标记";
  }
  return SAFETY_LABELS[level] ?? level;
}

function formatRelativeTime(isoText?: string | null): string {
  if (!isoText) {
    return "";
  }
  const value = new Date(isoText);
  if (Number.isNaN(value.getTime())) {
    return "";
  }
  return value.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatSummaryLabel(isoText: string, index: number): string {
  const value = new Date(isoText);
  if (Number.isNaN(value.getTime())) {
    return `小结版本 ${index + 1}`;
  }
  return `小结 ${value.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  })}`;
}

function reportNarrative(report: ReportParseResult | null, progress: ReportProgressPayload | null): string {
  if (!report) {
    return "上传体检报告后，这里会持续展示解析阶段、异常指标和后续可生成的小结。";
  }
  if (report.parse_status === "error") {
    return "解析失败，请重新上传更清晰的报告，或稍后再次尝试。";
  }
  if (report.parse_status === "needs_review") {
    return "报告已提取出初步结果，但仍建议人工复核关键字段。";
  }
  if (report.parse_status === "parsed") {
    return "解析已完成，可以围绕异常指标继续追问，并生成健康小结。";
  }
  return progress?.label ?? "报告已经进入解析队列，系统会持续更新当前阶段。";
}

function mapHistoryMessage(message: SessionMessage): MessageRow {
  // 历史消息只恢复当前工作台真正需要展示的字段，避免把后端对象直接泄漏到视图层。
  return {
    id: message.message_id,
    role: message.role === "assistant" ? "assistant" : "user",
    content: message.content,
    intent: message.intent,
    citations: message.citations,
    createdAt: message.created_at,
    streaming: false,
    statusLabel: null,
  };
}

export default function App() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageRow[]>([]);
  const [report, setReport] = useState<ReportParseResult | null>(null);
  const [reportProgress, setReportProgress] = useState<ReportProgressPayload | null>(null);
  const [summaryHistory, setSummaryHistory] = useState<SummaryArtifact[]>([]);
  const [selectedSummaryId, setSelectedSummaryId] = useState<string | null>(null);
  const [summaryOpen, setSummaryOpen] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [loadingSession, setLoadingSession] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [renamingSessionId, setRenamingSessionId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);

  const closeReportStreamRef = useRef<(() => void) | null>(null);
  const activeSessionIdRef = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const activeSession = sessions.find((item) => item.session_id === activeSessionId) ?? null;
  const reportReady = report ? ["parsed", "needs_review"].includes(report.parse_status) : true;
  const interactionLocked = busy !== null || loadingSession;
  const selectedSummary = useMemo(
    () => summaryHistory.find((item) => item.summary_id === selectedSummaryId) ?? summaryHistory[0] ?? null,
    [selectedSummaryId, summaryHistory],
  );

  const reportStateTone = reportTone(report?.parse_status);
  const abnormalCount = report?.abnormal_items.length ?? 0;
  const warningCount = report?.parse_warnings.length ?? 0;
  const progressValue = reportProgress?.progress ?? (report ? (reportReady || report.parse_status === "error" ? 100 : 5) : 0);
  const latestSummary = summaryHistory[0] ?? null;
  const summaryExcerpt =
    selectedSummary?.markdown
      .split("\n")
      .map((line) => line.trim())
      .find(Boolean) ?? "生成健康小结后，这里会展示最新版本的摘要预览。";

  const heroDescription = report
    ? "围绕已上传的体检报告、历史追问和健康小结持续推进当前会话。"
    : "先发起医学问题，或上传体检报告建立病例上下文，随后围绕异常指标继续追问。";
  const composerPlaceholder = report
    ? "例如：这份报告里总胆固醇偏高意味着什么？接下来建议做哪些复查？"
    : "例如：幽门螺杆菌阳性是什么意思？上传体检报告后还能围绕异常指标继续追问。";

  useEffect(() => {
    activeSessionIdRef.current = activeSessionId;
  }, [activeSessionId]);

  useEffect(() => {
    void bootstrapWorkspace();
    return () => {
      closeReportStreamRef.current?.();
      closeReportStreamRef.current = null;
    };
  }, []);

  function resetWorkspaceView() {
    // 切换会话前先清空右侧上下文状态，避免上一会话的报告和小结残留到新会话。
    closeReportStreamRef.current?.();
    closeReportStreamRef.current = null;
    setMessages([]);
    setReport(null);
    setReportProgress(null);
    setSummaryHistory([]);
    setSelectedSummaryId(null);
    setSummaryOpen(false);
  }

  function applySummaryHistory(nextSummaries: SummaryArtifact[]) {
    // 小结列表和默认选中项必须一起更新，避免弹窗预览和列表高亮不同步。
    setSummaryHistory(nextSummaries);
    setSelectedSummaryId(nextSummaries[0]?.summary_id ?? null);
  }

  async function bootstrapWorkspace() {
    // 首次进入页面时优先恢复会话工作台，没有历史会话时自动补一个空会话。
    setLoadingSession(true);
    setError(null);
    try {
      const loadedSessions = await listSessions();
      if (loadedSessions.length === 0) {
        const created = await createSession();
        setSessions([created]);
        await loadSessionWorkspace(created.session_id, [created]);
        return;
      }

      setSessions(loadedSessions);
      await loadSessionWorkspace(loadedSessions[0].session_id, loadedSessions);
    } catch (workspaceError) {
      setError(workspaceError instanceof Error ? workspaceError.message : "初始化会话失败。");
    } finally {
      setLoadingSession(false);
    }
  }

  async function refreshSessionList(preferredSessionId?: string): Promise<SessionSummary[]> {
    const loadedSessions = await listSessions();
    setSessions(loadedSessions);
    if (preferredSessionId && !loadedSessions.some((item) => item.session_id === preferredSessionId)) {
      setActiveSessionId(loadedSessions[0]?.session_id ?? null);
    }
    return loadedSessions;
  }

  async function loadSessionWorkspace(sessionId: string, cachedSessions?: SessionSummary[]) {
    // 进入某个会话时并行恢复：基础信息、消息历史、小结历史、报告状态。
    resetWorkspaceView();
    setLoadingSession(true);
    setError(null);
    setActiveSessionId(sessionId);

    try {
      if (!cachedSessions) {
        await refreshSessionList(sessionId);
      } else {
        setSessions(cachedSessions);
      }

      const [detail, history, summaries] = await Promise.all([
        getSessionDetail(sessionId),
        getSessionMessages(sessionId),
        getSessionSummaries(sessionId),
      ]);

      setMessages(history.map(mapHistoryMessage));
      applySummaryHistory(summaries);

      if (detail.report?.report_id) {
        const loadedReport = await getReport(detail.report.report_id);
        setReport(loadedReport);
        if (!["parsed", "needs_review", "error"].includes(loadedReport.parse_status)) {
          subscribeReportProgress(loadedReport.report_id, sessionId);
        }
      }
    } catch (sessionError) {
      resetWorkspaceView();
      setError(sessionError instanceof Error ? sessionError.message : "加载会话失败。");
    } finally {
      setLoadingSession(false);
    }
  }

  function subscribeReportProgress(reportId: string, sessionId: string) {
    // 报告解析流只服务当前激活会话，切走后立即断开，避免状态串到别的会话里。
    closeReportStreamRef.current?.();
    closeReportStreamRef.current = streamReportProgress(reportId, (event) => {
      if (activeSessionIdRef.current !== sessionId) {
        return;
      }

      if (event.event === "progress") {
        const data = getStreamEventData<ReportProgressPayload>(event);
        setReportProgress(data);
        setReport((previous) =>
          previous && previous.report_id === reportId ? { ...previous, parse_status: data.parse_status } : previous,
        );
        return;
      }

      if (event.event === "final") {
        const data = getStreamEventData<ReportParseResult>(event);
        setReport(data);
        setReportProgress({
          report_id: data.report_id,
          stage: data.parse_status === "error" ? "failed" : "completed",
          label: data.parse_status === "error" ? "报告解析失败" : "报告解析完成",
          progress: 100,
          parse_status: data.parse_status,
          done: true,
          error: null,
        });
        closeReportStreamRef.current?.();
        closeReportStreamRef.current = null;
        void refreshSessionList(sessionId);
        return;
      }

      if (event.event === "error") {
        const data = getStreamEventData<{ detail?: string }>(event);
        setError(data.detail ?? "报告进度流连接失败。");
      }
    });
  }

  async function handleCreateSession() {
    if (interactionLocked) {
      return;
    }

    setBusy("create-session");
    setError(null);
    try {
      const created = await createSession();
      const nextSessions = [created, ...sessions];
      setSessions(nextSessions);
      await loadSessionWorkspace(created.session_id, nextSessions);
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : "新建会话失败。");
    } finally {
      setBusy(null);
    }
  }

  async function handleSelectSession(sessionId: string) {
    if (interactionLocked || sessionId === activeSessionId) {
      return;
    }
    await loadSessionWorkspace(sessionId);
  }

  function startRename(sessionItem: SessionSummary) {
    setRenamingSessionId(sessionItem.session_id);
    setRenameDraft(sessionItem.title);
  }

  function cancelRename() {
    setRenamingSessionId(null);
    setRenameDraft("");
  }

  async function submitRename(sessionId: string) {
    const nextTitle = renameDraft.trim();
    if (!nextTitle) {
      setError("会话标题不能为空。");
      return;
    }

    setBusy("rename-session");
    setError(null);
    try {
      await renameSession(sessionId, nextTitle);
      await refreshSessionList(sessionId);
      cancelRename();
    } catch (renameError) {
      setError(renameError instanceof Error ? renameError.message : "重命名失败。");
    } finally {
      setBusy(null);
    }
  }

  async function handleDeleteSession(sessionId: string) {
    if (interactionLocked) {
      return;
    }

    const sessionItem = sessions.find((item) => item.session_id === sessionId);
    if (!sessionItem) {
      return;
    }

    const confirmed = window.confirm(`确认删除会话“${sessionItem.title}”吗？该会话的历史消息和健康小结会一并清理。`);
    if (!confirmed) {
      return;
    }

    setBusy("delete-session");
    setDeletingSessionId(sessionId);
    setError(null);
    try {
      await deleteSession(sessionId);

      if (renamingSessionId === sessionId) {
        cancelRename();
      }

      const remainingSessions = sessions.filter((item) => item.session_id !== sessionId);
      if (remainingSessions.length === 0) {
        const created = await createSession();
        setSessions([created]);
        await loadSessionWorkspace(created.session_id, [created]);
        return;
      }

      setSessions(remainingSessions);
      if (activeSessionId === sessionId) {
        const deletedIndex = sessions.findIndex((item) => item.session_id === sessionId);
        const fallbackSession =
          remainingSessions[deletedIndex] ?? remainingSessions[deletedIndex - 1] ?? remainingSessions[0];
        await loadSessionWorkspace(fallbackSession.session_id, remainingSessions);
      }
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "删除会话失败。");
    } finally {
      setDeletingSessionId(null);
      setBusy(null);
    }
  }

  function triggerReportUpload() {
    if (!activeSessionId || interactionLocked) {
      return;
    }
    fileInputRef.current?.click();
  }

  async function handleReportFileChange(event: ChangeEvent<HTMLInputElement>) {
    // 上传新报告时要同步清掉旧小结，因为后续小结必须绑定新的报告语境。
    const file = event.target.files?.[0] ?? null;
    event.target.value = "";
    if (!file) {
      return;
    }
    if (!activeSessionId) {
      setError("请先创建会话，再上传报告。");
      return;
    }

    setBusy("upload");
    setError(null);
    applySummaryHistory([]);
    setSummaryOpen(false);
    try {
      const result = await uploadReportToSession(activeSessionId, file);
      setReport(result);
      setReportProgress({
        report_id: result.report_id,
        stage: "queued",
        label: "上传完成，等待开始解析",
        progress: 5,
        parse_status: result.parse_status,
        done: false,
        error: null,
      });
      subscribeReportProgress(result.report_id, activeSessionId);
      await refreshSessionList(activeSessionId);
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "上传失败。");
    } finally {
      setBusy(null);
    }
  }

  async function handleChatSubmit(event: FormEvent<HTMLFormElement>) {
    // 聊天流用一个临时助手气泡承接状态、增量文本和最终结构化返回。
    event.preventDefault();
    if (!activeSessionId) {
      setError("请先创建会话。");
      return;
    }
    if (!prompt.trim()) {
      return;
    }

    const currentPrompt = prompt.trim();
    const sessionId = activeSessionId;
    const assistantId = crypto.randomUUID();
    setPrompt("");
    setMessages((previous) => [
      ...previous,
      { id: crypto.randomUUID(), role: "user", content: currentPrompt, createdAt: new Date().toISOString() },
      { id: assistantId, role: "assistant", content: "", statusLabel: "正在分析问题", streaming: true },
    ]);
    setBusy("chat");
    setError(null);

    try {
      await streamChat(
        {
          session_id: sessionId,
          report_id: report?.report_id ?? null,
          message: currentPrompt,
        },
        (event) => {
          if (activeSessionIdRef.current !== sessionId) {
            return;
          }

          if (event.event === "status") {
            const data = getStreamEventData<{ label: string }>(event);
            setMessages((previous) =>
              previous.map((item) =>
                item.id === assistantId ? { ...item, statusLabel: data.label, streaming: true } : item,
              ),
            );
            return;
          }

          if (event.event === "delta") {
            const data = getStreamEventData<{ text: string }>(event);
            setMessages((previous) =>
              previous.map((item) =>
                item.id === assistantId
                  ? { ...item, content: `${item.content}${data.text}`, statusLabel: "正在生成回答", streaming: true }
                  : item,
              ),
            );
            return;
          }

          if (event.event === "final") {
            const data = getStreamEventData<AgentResponse>(event);
            setMessages((previous) =>
              previous.map((item) =>
                item.id === assistantId
                  ? {
                      ...item,
                      content: data.answer,
                      intent: data.intent,
                      citations: data.citations,
                      meta: data,
                      statusLabel: null,
                      streaming: false,
                      createdAt: new Date().toISOString(),
                    }
                  : item,
              ),
            );
            return;
          }

          if (event.event === "error") {
            const data = getStreamEventData<{ detail?: string }>(event);
            setError(data.detail ?? "流式响应失败。");
          }
        },
      );

      await refreshSessionList(sessionId);
    } catch (chatError) {
      setError(chatError instanceof Error ? chatError.message : "对话请求失败。");
      setMessages((previous) =>
        previous.map((item) =>
          item.id === assistantId ? { ...item, statusLabel: null, streaming: false, content: item.content || "请求失败。" } : item,
        ),
      );
    } finally {
      setBusy(null);
    }
  }

  async function handleGenerateSummary() {
    // 小结生成依赖当前会话已经绑定的报告和对话上下文，所以这里直接走会话级接口。
    if (!activeSessionId) {
      setError("请先创建会话。");
      return;
    }
    if (!report) {
      setError("请先上传报告，再生成健康小结。");
      return;
    }
    if (!reportReady) {
      setError("报告仍在解析中，请稍后再生成健康小结。");
      return;
    }

    setBusy("summary");
    setError(null);
    try {
      const result = await generateSummaryForSession(activeSessionId);
      const nextHistory = [result, ...summaryHistory.filter((item) => item.summary_id !== result.summary_id)];
      setSummaryHistory(nextHistory);
      setSelectedSummaryId(result.summary_id);
      setSummaryOpen(true);
    } catch (summaryError) {
      setError(summaryError instanceof Error ? summaryError.message : "健康小结生成失败。");
    } finally {
      setBusy(null);
    }
  }

  return (
    <main className="workspace-shell">
      <aside className="session-sidebar">
        <div className="sidebar-brand">
          <h1>历史会话</h1>
        </div>

        <button className="primary-button sidebar-create-button" type="button" onClick={handleCreateSession} disabled={interactionLocked}>
          {busy === "create-session" ? "正在创建..." : "新建会话"}
        </button>

        <div className="session-list">
          {sessions.map((sessionItem) => {
            const selected = sessionItem.session_id === activeSessionId;
            const renaming = sessionItem.session_id === renamingSessionId;
            const deleting = sessionItem.session_id === deletingSessionId;

            return (
              <div className={`session-card ${selected ? "active" : ""}`} key={sessionItem.session_id}>
                <button
                  className="session-select-button"
                  type="button"
                  onClick={() => void handleSelectSession(sessionItem.session_id)}
                  disabled={interactionLocked}
                >
                  <div className="session-card-top">
                    {renaming ? (
                      <input
                        className="session-rename-input"
                        value={renameDraft}
                        onChange={(event) => setRenameDraft(event.target.value)}
                        onClick={(event) => event.stopPropagation()}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") {
                            event.preventDefault();
                            void submitRename(sessionItem.session_id);
                          }
                          if (event.key === "Escape") {
                            event.preventDefault();
                            cancelRename();
                          }
                        }}
                      />
                    ) : (
                      <>
                        <strong className="session-title">{sessionItem.title}</strong>
                        {sessionItem.report ? <span className="session-inline-chip">已附报告</span> : null}
                      </>
                    )}
                  </div>

                  <span className="session-preview">{sessionItem.last_message_preview || "暂无消息，等待新的问题或报告。"}</span>

                  <span className="session-meta">
                    <span>{formatRelativeTime(sessionItem.last_message_at) || "刚刚创建"}</span>
                    <span>{sessionItem.message_count} 条消息</span>
                  </span>
                </button>

                <div className="session-actions">
                  {renaming ? (
                    <>
                      <button className="text-button" type="button" onClick={() => void submitRename(sessionItem.session_id)} disabled={interactionLocked}>
                        保存
                      </button>
                      <button className="text-button subtle" type="button" onClick={cancelRename} disabled={interactionLocked}>
                        取消
                      </button>
                    </>
                  ) : (
                    <>
                      <button className="text-button subtle" type="button" onClick={() => startRename(sessionItem)} disabled={interactionLocked}>
                        重命名
                      </button>
                      <button className="text-button danger" type="button" onClick={() => void handleDeleteSession(sessionItem.session_id)} disabled={interactionLocked}>
                        {deleting ? "删除中..." : "删除"}
                      </button>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </aside>

      <section className="workspace-main">
        {error ? <div className="error-banner">{error}</div> : null}

        <section className="conversation-panel">
          <header className="workspace-hero">
            <div className="workspace-hero-copy">
              <p className="eyebrow">Medical Workspace</p>
              <h2>{activeSession?.title ?? "健康咨询会话"}</h2>
              <p className="workspace-hero-description">{heroDescription}</p>
            </div>
            <div className="panel-heading-meta">
              <span className={`status-pill tone-${reportStateTone}`}>{report ? reportStatusText(report.parse_status) : "未绑定报告"}</span>
              {summaryHistory.length > 0 ? <span className="status-pill tone-ready">已有健康小结</span> : null}
            </div>
          </header>

          <div className="workspace-context-bar">
            <span className="context-pill">当前会话：{activeSession?.title ?? "未命名会话"}</span>
            <span className="context-pill">消息 {activeSession?.message_count ?? 0}</span>
            {report ? <span className="context-pill">异常指标 {abnormalCount}</span> : null}
            {reportProgress ? <span className="context-pill">解析进度 {progressValue}%</span> : null}
            {summaryHistory.length > 0 ? <span className="context-pill">小结 {summaryHistory.length} 份</span> : null}
          </div>

          <div className="chat-window">
            {loadingSession ? <p className="muted">正在加载会话...</p> : null}

            {!loadingSession && messages.length === 0 ? (
              <div className="empty-chat-state">
                <p className="section-label">Medical Agent</p>
                <h3>从一个问题开始，或先上传体检报告</h3>
                <p>{heroDescription}</p>
                <div className="empty-chat-actions">
                  <button className="secondary-button" type="button" onClick={triggerReportUpload} disabled={!activeSessionId || interactionLocked}>
                    上传体检报告
                  </button>
                  <button className="secondary-button" type="button" onClick={handleGenerateSummary} disabled={interactionLocked || !report || !reportReady}>
                    生成健康小结
                  </button>
                </div>
              </div>
            ) : null}

            {messages.map((message) => (
              <article className={`message ${message.role}`} key={message.id}>
                <div className="message-header">
                  <span className="message-role">{message.role === "assistant" ? "医疗助手" : "你"}</span>
                  {message.createdAt ? <span className="message-time">{formatRelativeTime(message.createdAt)}</span> : null}
                </div>

                <div className="message-bubble">
                  {message.role === "assistant" ? (
                    <div className="message-body rich">
                      {renderMessageContent(message.content)}
                      {message.streaming ? (
                        <div className="typing-line" aria-live="polite">
                          <span className="typing-cursor" aria-hidden="true" />
                          <span className="typing-text">正在生成</span>
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <div className="message-body">{message.content}</div>
                  )}
                </div>

                {message.statusLabel ? (
                  <div className="stream-status">
                    <span className="stream-status-dot" aria-hidden="true" />
                    <span>{message.statusLabel}</span>
                  </div>
                ) : null}

                {message.meta ? (
                  <div className="message-footer">
                    <div className="chip-row">
                      <span className="meta-chip">意图：{intentText(message.meta.intent)}</span>
                      <span className="meta-chip">安全级别：{safetyText(message.meta.safety_level)}</span>
                      {message.meta.used_tools.map((tool) => (
                        <span className="meta-chip" key={`${message.id}-${tool}`}>
                          工具：{tool}
                        </span>
                      ))}
                    </div>

                    {message.meta.citations.length > 0 ? (
                      <div className="chip-row">
                        {message.meta.citations.map((citation) =>
                          citation.url ? (
                            <a className="source-chip" key={`${message.id}-${citation.doc_id}`} href={citation.url} target="_blank" rel="noreferrer">
                              [{citation.trust_tier}] {citation.title}
                            </a>
                          ) : (
                            <span className="source-chip" key={`${message.id}-${citation.doc_id}`}>
                              [{citation.trust_tier}] {citation.title}
                            </span>
                          ),
                        )}
                      </div>
                    ) : null}

                    {message.meta.follow_up_questions.length > 0 ? (
                      <div className="follow-up-block">
                        <strong>建议继续追问</strong>
                        <ul className="message-list">
                          {message.meta.follow_up_questions.map((item, index) => (
                            <li key={`${message.id}-follow-up-${index}`}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}

                    {message.meta.debug ? (
                      <details className="debug-panel">
                        <summary>调试信息</summary>
                        <div className="debug-grid">
                          <section className="debug-section">
                            <h6>used_tools</h6>
                            <pre className="debug-pre">{formatDebugValue(message.meta.used_tools)}</pre>
                          </section>
                          {message.meta.debug.analysis ? (
                            <section className="debug-section">
                              <h6>analysis</h6>
                              <pre className="debug-pre">{formatDebugValue(message.meta.debug.analysis)}</pre>
                            </section>
                          ) : null}
                          {message.meta.debug.plan ? (
                            <section className="debug-section">
                              <h6>plan</h6>
                              <pre className="debug-pre">{formatDebugValue(message.meta.debug.plan)}</pre>
                            </section>
                          ) : null}
                          {message.meta.debug.synthesis ? (
                            <section className="debug-section">
                              <h6>synthesis</h6>
                              <pre className="debug-pre">{formatDebugValue(message.meta.debug.synthesis)}</pre>
                            </section>
                          ) : null}
                        </div>
                      </details>
                    ) : null}
                  </div>
                ) : message.citations && message.citations.length > 0 ? (
                  <div className="message-footer">
                    <div className="chip-row">
                      {message.intent ? <span className="meta-chip">意图：{intentText(message.intent)}</span> : null}
                      {message.citations.map((citation) =>
                        citation.url ? (
                          <a className="source-chip" key={`${message.id}-${citation.doc_id}`} href={citation.url} target="_blank" rel="noreferrer">
                            [{citation.trust_tier}] {citation.title}
                          </a>
                        ) : (
                          <span className="source-chip" key={`${message.id}-${citation.doc_id}`}>
                            [{citation.trust_tier}] {citation.title}
                          </span>
                        ),
                      )}
                    </div>
                  </div>
                ) : null}
              </article>
            ))}
          </div>

          <div className="assistant-context">
            <section className="compact-panel">
              <div className="compact-panel-head">
                <div>
                  <p className="section-label">Report</p>
                  <h3>报告上下文</h3>
                </div>
                <span className={`status-pill tone-${reportStateTone}`}>{report ? reportStatusText(report.parse_status) : "未上传"}</span>
              </div>
              {report ? (
                <>
                  <p className="compact-panel-text">{reportNarrative(report, reportProgress)}</p>
                  <div className="progress-track compact" aria-hidden="true">
                    <span className="progress-fill" style={{ width: `${Math.max(progressValue, 8)}%` }} />
                  </div>
                  <div className="compact-metrics">
                    <span className="context-pill">{report.file_name}</span>
                    <span className="context-pill">异常 {abnormalCount}</span>
                    <span className="context-pill">提示 {warningCount}</span>
                  </div>
                </>
              ) : (
                <p className="compact-panel-text">上传体检报告后，后续追问和小结会自动沿用当前报告上下文。</p>
              )}
            </section>

            <section className="compact-panel">
              <div className="compact-panel-head">
                <div>
                  <p className="section-label">Summary</p>
                  <h3>健康小结</h3>
                </div>
                <span className="status-pill tone-ready">{summaryHistory.length} 份</span>
              </div>
              <p className="compact-panel-text">
                {summaryHistory.length > 0
                  ? `最新版本生成于 ${formatRelativeTime(latestSummary?.created_at)}。`
                  : "生成健康小结后，可在这里快速预览最新版本并下载 PDF。"}
              </p>
              <div className="compact-metrics">
                <span className="context-pill">{latestSummary ? formatSummaryLabel(latestSummary.created_at, 0) : "尚未生成"}</span>
                <button className="secondary-button" type="button" onClick={() => setSummaryOpen(true)} disabled={summaryHistory.length === 0}>
                  查看历史
                </button>
              </div>
            </section>
          </div>

          <form className="composer-shell" onSubmit={handleChatSubmit}>
            <label className="composer-label" htmlFor="composer-input">
              输入问题
            </label>

            <textarea
              id="composer-input"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder={composerPlaceholder}
              disabled={interactionLocked || (!reportReady && report !== null)}
            />

            <div className="composer-toolbar">
              <div className="composer-toolbar-left">
                <input
                  ref={fileInputRef}
                  className="visually-hidden"
                  type="file"
                  accept=".pdf,.png,.jpg,.jpeg"
                  onChange={handleReportFileChange}
                />
                <button className="secondary-button" type="button" onClick={triggerReportUpload} disabled={!activeSessionId || interactionLocked}>
                  {report ? "更换体检报告" : "上传体检报告"}
                </button>
                <button className="secondary-button" type="button" onClick={handleGenerateSummary} disabled={interactionLocked || !report || !reportReady}>
                  {busy === "summary" ? "生成中..." : "生成健康小结"}
                </button>
              </div>

              <div className="composer-toolbar-right">
                <button className="secondary-button" type="button" onClick={() => setSummaryOpen(true)} disabled={summaryHistory.length === 0}>
                  查看小结
                </button>
                <button className="primary-button" type="submit" disabled={interactionLocked || (!reportReady && report !== null)}>
                  {busy === "chat" ? "处理中..." : "发送"}
                </button>
              </div>
            </div>
          </form>
        </section>
      </section>

      {summaryOpen ? (
        <div className="summary-modal-backdrop" role="presentation" onClick={() => setSummaryOpen(false)}>
          <section className="summary-modal" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
            <div className="summary-modal-header">
              <div>
                <p className="eyebrow">Summary Archive</p>
                <h3>健康小结</h3>
              </div>
              <button className="text-button subtle" type="button" onClick={() => setSummaryOpen(false)}>
                关闭
              </button>
            </div>

            <div className="summary-modal-content">
              <aside className="summary-history">
                <div className="summary-history-header">
                  <strong>历史版本</strong>
                  <span>{summaryHistory.length} 份</span>
                </div>

                {summaryHistory.length === 0 ? (
                  <div className="summary-history-empty">当前会话还没有生成过健康小结。</div>
                ) : (
                  <div className="summary-history-list">
                    {summaryHistory.map((item, index) => {
                      const selected = item.summary_id === selectedSummary?.summary_id;
                      return (
                        <button
                          key={item.summary_id}
                          className={`summary-history-item ${selected ? "active" : ""}`}
                          type="button"
                          onClick={() => setSelectedSummaryId(item.summary_id)}
                        >
                          <strong>{formatSummaryLabel(item.created_at, index)}</strong>
                          <span>{formatRelativeTime(item.created_at)}</span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </aside>

              <section className="summary-preview-panel">
                {selectedSummary ? (
                  <>
                    <div className="summary-preview-meta">
                      <span className="status-pill tone-ready">生成时间：{formatRelativeTime(selectedSummary.created_at)}</span>
                    </div>
                    <article className="markdown-preview">{renderMessageContent(selectedSummary.markdown)}</article>
                  </>
                ) : (
                  <div className="summary-history-empty">请选择一份健康小结进行预览。</div>
                )}
              </section>
            </div>

            <div className="summary-modal-actions">
              {selectedSummary ? (
                <a
                  className="primary-button link-button"
                  href={`${API_BASE}/summaries/${selectedSummary.summary_id}/pdf`}
                  target="_blank"
                  rel="noreferrer"
                >
                  下载 PDF
                </a>
              ) : null}
            </div>
          </section>
        </div>
      ) : null}
    </main>
  );
}
