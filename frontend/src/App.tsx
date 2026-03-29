import { Fragment, FormEvent, ReactNode, useEffect, useRef, useState } from "react";

import { generateSummary, streamChat, streamReportProgress, uploadReport } from "./api";
import type {
  AgentResponse,
  ChatStreamEvent,
  ReportParseResult,
  ReportProgressPayload,
  SummaryArtifact,
} from "./types";

type MessageRow = {
  id: string;
  role: "user" | "assistant";
  content: string;
  // meta 只在助手消息里出现，用来挂接 Agent 返回的结构化信息，
  // 例如 intent / used_tools / citations / debug。
  meta?: AgentResponse;
  statusLabel?: string | null;
  streaming?: boolean;
};

function renderInline(text: string): ReactNode[] {
  // 这里只做了最轻量的富文本支持：处理 **加粗**。
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

function renderMessageContent(content: string): ReactNode {
  // 这不是完整 Markdown 解析器，而是一个够用的简化版。
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
  if (status === "processing") return "解析中";
  if (status === "parsed") return "已完成";
  if (status === "needs_review") return "需复核";
  if (status === "error") return "失败";
  return status;
}

export default function App() {
  // report: 当前上传并正在分析/已分析完成的报告
  const [report, setReport] = useState<ReportParseResult | null>(null);
  // reportProgress: 解析过程中的实时阶段和百分比
  const [reportProgress, setReportProgress] = useState<ReportProgressPayload | null>(null);
  // summary: 最终生成的小结 Markdown + PDF
  const [summary, setSummary] = useState<SummaryArtifact | null>(null);
  // messages: 对话区的完整消息列表
  const [messages, setMessages] = useState<MessageRow[]>([]);
  // sessionId: 后端会话 id，用来把多轮对话串起来
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  // busy: 当前页面正在执行什么操作，用于禁用按钮和显示处理中状态
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // EventSource 需要在重新订阅或组件卸载时关闭，否则会残留旧连接。
  const closeReportStreamRef = useRef<(() => void) | null>(null);

  // 报告只有在解析完成或需要人工复核时，才允许继续围绕它提问或生成小结。
  const reportReady = report ? ["parsed", "needs_review"].includes(report.parse_status) : true;

  useEffect(() => {
    return () => {
      closeReportStreamRef.current?.();
      closeReportStreamRef.current = null;
    };
  }, []);

  function subscribeReportProgress(reportId: string) {
    // 同一时刻只保留一条报告进度流，避免切换报告后多个 EventSource 同时更新 UI。
    closeReportStreamRef.current?.();
    closeReportStreamRef.current = streamReportProgress(reportId, (event) => {
      if (event.event === "progress") {
        const data = getStreamEventData<ReportProgressPayload>(event);
        setReportProgress(data);
        setReport((prev) => (prev && prev.report_id === reportId ? { ...prev, parse_status: data.parse_status } : prev));
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
        return;
      }

      if (event.event === "error") {
        const data = getStreamEventData<{ detail?: string }>(event);
        setError(data.detail ?? "报告进度流连接失败。");
      }
    });
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    // 上传流程：
    // 1. 上传文件
    // 2. 拿到 report_id
    // 3. 立刻订阅后台解析进度
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const file = form.get("report") as File | null;
    if (!file || file.size === 0) {
      setError("请先选择报告文件。");
      return;
    }

    setBusy("upload");
    setError(null);
    setSummary(null);
    try {
      const result = await uploadReport(file);
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
      subscribeReportProgress(result.report_id);
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "上传失败。");
    } finally {
      setBusy(null);
    }
  }

  async function handleChat(event: FormEvent<HTMLFormElement>) {
    // 聊天发送时，会先插入：
    // - 一条用户消息
    // - 一条空的助手占位消息
    // 然后随着 SSE delta 不断往助手消息里追加内容。
    event.preventDefault();
    if (!prompt.trim()) {
      return;
    }

    const currentPrompt = prompt.trim();
    const assistantId = crypto.randomUUID();
    setPrompt("");
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", content: currentPrompt },
      { id: assistantId, role: "assistant", content: "", statusLabel: "分析问题中", streaming: true },
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
          if (event.event === "session") {
            // 后端第一次返回 session_id 时，前端要记住它，
            // 后续追问才能共享同一段短期上下文。
            const data = getStreamEventData<{ session_id: string }>(event);
            setSessionId(data.session_id);
            return;
          }

          if (event.event === "status") {
            // status 不是正文，而是“当前 Agent 处理到哪一步了”。
            const data = getStreamEventData<{ label: string }>(event);
            setMessages((prev) =>
              prev.map((item) => (item.id === assistantId ? { ...item, statusLabel: data.label, streaming: true } : item)),
            );
            return;
          }

          if (event.event === "delta") {
            // delta 是流式正文增量，直接拼接到当前助手消息尾部。
            const data = getStreamEventData<{ text: string }>(event);
            setMessages((prev) =>
              prev.map((item) =>
                item.id === assistantId
                  ? { ...item, content: `${item.content}${data.text}`, statusLabel: "正在生成回答", streaming: true }
                  : item,
              ),
            );
            return;
          }

          if (event.event === "final") {
            // final 是最终结构化结果，会覆盖流式草稿，
            // 同时补上 meta / debug / citations 等信息。
            const data = getStreamEventData<AgentResponse>(event);
            setSessionId(data.session_id);
            setMessages((prev) =>
              prev.map((item) =>
                item.id === assistantId
                  ? { ...item, content: data.answer, meta: data, statusLabel: null, streaming: false }
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
    } catch (chatError) {
      setError(chatError instanceof Error ? chatError.message : "对话请求失败。");
      setMessages((prev) =>
        prev.map((item) =>
          item.id === assistantId ? { ...item, statusLabel: null, streaming: false, content: item.content || "请求失败。" } : item,
        ),
      );
    } finally {
      setBusy(null);
    }
  }

  async function handleGenerateSummary() {
    // 小结生成依赖两样东西：
    // 1. 已有报告
    // 2. 一个稳定的 session_id
    if (!report) {
      setError("请先上传报告，再生成健康小结。");
      return;
    }

    const activeSessionId = sessionId ?? crypto.randomUUID();
    setSessionId(activeSessionId);
    setBusy("summary");
    setError(null);
    try {
      const result = await generateSummary({
        session_id: activeSessionId,
        report_id: report.report_id,
      });
      setSummary(result);
    } catch (summaryError) {
      setError(summaryError instanceof Error ? summaryError.message : "健康小结生成失败。");
    } finally {
      setBusy(null);
    }
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Medical Agent v1</p>
          <h1>体检报告解读与健康咨询工作台</h1>
          <p className="hero-copy">
            上传报告后系统会自动解析并持续显示进度。对话区支持流式输出，适合围绕指标异常、术语解释和复查建议继续追问。
          </p>
        </div>
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="grid">
        <article className="panel">
          <div className="panel-header">
            <h2>1. 报告上传</h2>
            <span>{report ? report.file_name : "尚未上传"}</span>
          </div>
          <form onSubmit={handleUpload} className="stack">
            <input name="report" type="file" accept=".pdf,.png,.jpg,.jpeg" />
            <button className="primary-button" type="submit" disabled={busy !== null}>
              {busy === "upload" ? "上传中..." : "上传并开始解析"}
            </button>
          </form>

          {report ? (
            <div className="stack">
              <div className="status-line">
                <strong>解析状态</strong>
                <span>{reportStatusText(report.parse_status)}</span>
              </div>
              {reportProgress ? (
                <div className="progress-card">
                  <div className="progress-head">
                    <span className="progress-stage">{reportProgress.label}</span>
                    <strong>{reportProgress.progress}%</strong>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-bar-fill" style={{ width: `${reportProgress.progress}%` }} />
                  </div>
                  <div className="progress-meta">
                    <span>阶段：{reportProgress.stage}</span>
                    <span>{reportProgress.done ? "已结束" : "进行中"}</span>
                  </div>
                </div>
              ) : null}
              {report.parse_warnings.length > 0 ? (
                <div className="warning-box">
                  {report.parse_warnings.map((warning) => (
                    <p key={warning}>{warning}</p>
                  ))}
                </div>
              ) : null}
              <div className="metric-list">
                {report.abnormal_items.length > 0 ? (
                  report.abnormal_items.map((item) => (
                    <div className={`metric-card ${item.status}`} key={`${item.name}-${item.value_raw}`}>
                      <strong>{item.name}</strong>
                      <span>
                        {item.value_raw}
                        {item.unit}
                      </span>
                      <small>{item.reference_range || "无参考范围"}</small>
                    </div>
                  ))
                ) : (
                  <p className="muted">{reportReady ? "暂未识别到明确异常项。" : "正在解析报告，异常项会在完成后自动显示。"}</p>
                )}
              </div>
            </div>
          ) : null}
        </article>

        <article className="panel">
          <div className="panel-header">
            <h2>2. 健康咨询</h2>
            <span>{messages.length} 条消息</span>
          </div>
          <div className="chat-window">
            {messages.length === 0 ? <p className="muted">可以询问报告指标、医学术语、常见症状方向或复查建议。</p> : null}
            {messages.map((message) => (
              <div className={`message ${message.role}`} key={message.id}>
                {message.role === "assistant" ? (
                  <div className="message-body rich">
                    {/* 助手消息支持轻量富文本渲染，便于阅读结构化回答。 */}
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

                {message.statusLabel ? (
                  <div className="stream-status">
                    <span className="stream-status-dot" aria-hidden="true" />
                    <span>{message.statusLabel}</span>
                  </div>
                ) : null}

                {message.meta ? (
                  <div className="message-footer">
                    <div className="chip-row">
                      <span className="meta-chip">意图：{message.meta.intent}</span>
                      {message.meta.used_tools.map((tool) => (
                        <span className="meta-chip" key={`${message.id}-${tool}`}>
                          工具：{tool}
                        </span>
                      ))}
                    </div>
                    {message.meta.citations.length > 0 ? (
                      <div className="chip-row">
                        {message.meta.citations.map((citation) => (
                          <span className="source-chip" key={citation.doc_id}>
                            [{citation.trust_tier}] {citation.title}
                          </span>
                        ))}
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
                ) : null}
              </div>
            ))}
          </div>

          <form onSubmit={handleChat} className="chat-form">
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="例如：这些异常指标代表了什么？总胆固醇和 LDL-C 偏高说明什么？"
              disabled={!reportReady && report !== null}
            />
            <button className="primary-button" type="submit" disabled={busy !== null || (!reportReady && report !== null)}>
              {busy === "chat" ? "处理中..." : "发送"}
            </button>
          </form>
        </article>

        <article className="panel">
          <div className="panel-header">
            <h2>3. 健康小结</h2>
            <span>{summary ? "已生成" : "待生成"}</span>
          </div>
          <button className="primary-button" onClick={handleGenerateSummary} disabled={busy !== null || !report || !reportReady}>
            {busy === "summary" ? "生成中..." : "生成 Markdown / PDF"}
          </button>
          {summary ? (
            <div className="stack">
              <pre className="markdown-preview">{summary.markdown}</pre>
              <a
                className="secondary-button link-button"
                href={`http://localhost:8000/api/summaries/${summary.summary_id}/pdf`}
                target="_blank"
                rel="noreferrer"
              >
                下载 PDF
              </a>
            </div>
          ) : (
            <p className="muted">生成后会在这里显示 Markdown 预览，并提供 PDF 下载。</p>
          )}
        </article>
      </section>
    </main>
  );
}
