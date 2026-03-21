import { FormEvent, useEffect, useState } from "react";

import { bootstrapKnowledge, generateSummary, getKnowledgeSources, sendChat, uploadReport } from "./api";
import type { AgentResponse, KnowledgeSourcesResponse, ReportParseResult, SummaryArtifact } from "./types";

type MessageRow = {
  role: "user" | "assistant";
  content: string;
  meta?: AgentResponse;
};

export default function App() {
  const [report, setReport] = useState<ReportParseResult | null>(null);
  const [summary, setSummary] = useState<SummaryArtifact | null>(null);
  const [knowledgeStats, setKnowledgeStats] = useState<KnowledgeSourcesResponse | null>(null);
  const [messages, setMessages] = useState<MessageRow[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void refreshKnowledge();
  }, []);

  async function refreshKnowledge() {
    try {
      setKnowledgeStats(await getKnowledgeSources());
    } catch {
      setKnowledgeStats(null);
    }
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const file = form.get("report") as File | null;
    if (!file || file.size === 0) {
      setError("请先选择报告文件。");
      return;
    }
    setBusy("upload");
    setError(null);
    try {
      const result = await uploadReport(file);
      setReport(result);
      setSummary(null);
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "上传失败。");
    } finally {
      setBusy(null);
    }
  }

  async function handleChat(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!prompt.trim()) {
      return;
    }
    const currentPrompt = prompt.trim();
    setPrompt("");
    setMessages((prev) => [...prev, { role: "user", content: currentPrompt }]);
    setBusy("chat");
    setError(null);
    try {
      const response = await sendChat({
        session_id: sessionId,
        report_id: report?.report_id ?? null,
        message: currentPrompt,
      });
      setSessionId(response.session_id);
      setMessages((prev) => [...prev, { role: "assistant", content: response.answer, meta: response }]);
    } catch (chatError) {
      setError(chatError instanceof Error ? chatError.message : "对话请求失败。");
    } finally {
      setBusy(null);
    }
  }

  async function handleBootstrapKnowledge() {
    setBusy("knowledge");
    setError(null);
    try {
      await bootstrapKnowledge();
      await refreshKnowledge();
    } catch (bootstrapError) {
      setError(bootstrapError instanceof Error ? bootstrapError.message : "知识库初始化失败。");
    } finally {
      setBusy(null);
    }
  }

  async function handleGenerateSummary() {
    if (!report) {
      setError("请先上传报告再生成小结。");
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
      setError(summaryError instanceof Error ? summaryError.message : "小结生成失败。");
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
            上传中文体检或检验报告，查看结构化指标，继续追问异常项，并生成带来源引用的健康小结。
          </p>
        </div>
        <button className="secondary-button" onClick={handleBootstrapKnowledge} disabled={busy !== null}>
          {busy === "knowledge" ? "初始化中..." : "初始化联网知识库"}
        </button>
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
              {busy === "upload" ? "解析中..." : "上传并解析"}
            </button>
          </form>
          {report ? (
            <div className="stack">
              <div className="status-line">
                <strong>状态</strong>
                <span>{report.parse_status}</span>
              </div>
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
                  <p className="muted">暂未识别出明确异常项。</p>
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
            {messages.length === 0 ? <p className="muted">可以询问报告指标、医学术语或常见症状方向。</p> : null}
            {messages.map((message, index) => (
              <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
                <p>{message.content}</p>
                {message.meta?.citations?.length ? (
                  <div className="citation-list">
                    {message.meta.citations.map((citation) => (
                      <a href={citation.url} key={citation.doc_id} target="_blank" rel="noreferrer">
                        [{citation.trust_tier}] {citation.title}
                      </a>
                    ))}
                  </div>
                ) : null}
              </div>
            ))}
          </div>
          <form onSubmit={handleChat} className="chat-form">
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="例如：报告里的甘油三酯偏高代表什么？需要挂什么科？"
            />
            <button className="primary-button" type="submit" disabled={busy !== null}>
              {busy === "chat" ? "分析中..." : "发送"}
            </button>
          </form>
        </article>

        <article className="panel">
          <div className="panel-header">
            <h2>3. 健康小结</h2>
            <span>{summary ? "已生成" : "待生成"}</span>
          </div>
          <button className="primary-button" onClick={handleGenerateSummary} disabled={busy !== null || !report}>
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
            <p className="muted">生成后会在这里展示结构化小结，并提供 PDF 下载。</p>
          )}
        </article>

        <article className="panel">
          <div className="panel-header">
            <h2>4. 知识源状态</h2>
            <span>{knowledgeStats ? `${knowledgeStats.total_docs} 条` : "不可用"}</span>
          </div>
          {knowledgeStats ? (
            <div className="stack">
              <div className="trust-grid">
                {Object.entries(knowledgeStats.trust_breakdown).map(([tier, count]) => (
                  <div className="trust-card" key={tier}>
                    <strong>{tier}</strong>
                    <span>{count}</span>
                  </div>
                ))}
              </div>
              <div className="source-list">
                {knowledgeStats.recent_docs.map((doc) => (
                  <a href={doc.url} key={doc.doc_id} target="_blank" rel="noreferrer">
                    <span>[{doc.trust_tier}]</span>
                    <strong>{doc.title}</strong>
                    <small>{doc.source_domain}</small>
                  </a>
                ))}
              </div>
            </div>
          ) : (
            <p className="muted">启动后端后，这里会显示知识源统计和信任分级。</p>
          )}
        </article>
      </section>
    </main>
  );
}
