import type {
  AgentRunDetail,
  AgentTaskRunSummary,
  ChatStreamEvent,
  KnowledgeSourcesResponse,
  ReportParseResult,
  SessionDetail,
  SessionMessage,
  SessionSummary,
  SummaryArtifact,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api";

async function unwrap<T>(response: Response): Promise<T> {
  // 普通 JSON 接口统一在这里做错误处理，避免每个 API 都重复写样板代码。
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "请求失败。" }));
    throw new Error(error.detail ?? "请求失败。");
  }
  return (await response.json()) as T;
}

function parseSseBlock(rawEvent: string): ChatStreamEvent | null {
  // 后端返回的是标准 SSE 文本块，这里将其转换为前端更容易消费的结构。
  const lines = rawEvent.split("\n").filter(Boolean);
  let eventName = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  if (!dataLines.length) {
    return null;
  }

  return {
    event: eventName,
    data: JSON.parse(dataLines.join("\n")),
  };
}

export async function uploadReport(file: File): Promise<ReportParseResult> {
  // 通用上传接口保留给非会话场景使用。
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/reports/upload`, {
    method: "POST",
    body: formData,
  });
  return unwrap<ReportParseResult>(response);
}

export async function getReport(reportId: string): Promise<ReportParseResult> {
  const response = await fetch(`${API_BASE}/reports/${reportId}`);
  return unwrap<ReportParseResult>(response);
}

export async function uploadReportToSession(sessionId: string, file: File): Promise<ReportParseResult> {
  // 会话化页面优先走这个接口，避免前端额外做“上传后再绑定”的二次调用。
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/reports`, {
    method: "POST",
    body: formData,
  });
  return unwrap<ReportParseResult>(response);
}

export async function streamChat(
  payload: {
    session_id?: string | null;
    report_id?: string | null;
    message: string;
  },
  onEvent: (event: ChatStreamEvent) => void,
): Promise<void> {
  // 聊天走 POST，所以用 fetch + ReadableStream 手动解析 SSE。
  const response = await fetch(`${API_BASE}/agent/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    const fallback = await response.json().catch(() => ({ detail: "流式请求失败。" }));
    throw new Error(fallback.detail ?? "流式请求失败。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    let boundaryIndex = buffer.indexOf("\n\n");
    while (boundaryIndex !== -1) {
      const rawEvent = buffer.slice(0, boundaryIndex);
      buffer = buffer.slice(boundaryIndex + 2);
      boundaryIndex = buffer.indexOf("\n\n");

      const parsed = parseSseBlock(rawEvent);
      if (parsed) {
        onEvent(parsed);
      }
    }
  }
}

export function streamReportProgress(reportId: string, onEvent: (event: ChatStreamEvent) => void): () => void {
  // 报告进度是 GET 场景，直接使用浏览器原生 EventSource 即可。
  const eventSource = new EventSource(`${API_BASE}/reports/${reportId}/stream`);

  eventSource.addEventListener("progress", (event) => {
    onEvent({
      event: "progress",
      data: JSON.parse((event as MessageEvent<string>).data),
    });
  });

  eventSource.addEventListener("final", (event) => {
    onEvent({
      event: "final",
      data: JSON.parse((event as MessageEvent<string>).data),
    });
    eventSource.close();
  });

  eventSource.addEventListener("error", () => {
    onEvent({
      event: "error",
      data: { detail: "报告进度流连接已中断。" },
    });
    eventSource.close();
  });

  return () => eventSource.close();
}

export async function generateSummary(payload: {
  session_id: string;
  report_id: string;
}): Promise<SummaryArtifact> {
  const response = await fetch(`${API_BASE}/summaries/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return unwrap<SummaryArtifact>(response);
}

export async function generateSummaryForSession(sessionId: string): Promise<SummaryArtifact> {
  // 当前页面的小结由后端基于会话上下文自动收集材料并生成。
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/summaries/generate`, {
    method: "POST",
  });
  return unwrap<SummaryArtifact>(response);
}

export async function getLatestSummary(sessionId: string): Promise<SummaryArtifact | null> {
  // 切回历史会话时，如果此前已经生成过小结，就直接恢复最近一份。
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/summaries/latest`);
  if (response.status === 404) {
    return null;
  }
  return unwrap<SummaryArtifact>(response);
}

export async function getSessionSummaries(sessionId: string): Promise<SummaryArtifact[]> {
  // 小结历史列表用于在弹窗内切换查看不同版本的小结内容。
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/summaries`);
  return unwrap<SummaryArtifact[]>(response);
}

export async function listSessions(): Promise<SessionSummary[]> {
  const response = await fetch(`${API_BASE}/sessions`);
  return unwrap<SessionSummary[]>(response);
}

export async function createSession(title?: string): Promise<SessionSummary> {
  const response = await fetch(`${API_BASE}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: title ?? null }),
  });
  return unwrap<SessionSummary>(response);
}

export async function getSessionDetail(sessionId: string): Promise<SessionDetail> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}`);
  return unwrap<SessionDetail>(response);
}

export async function renameSession(sessionId: string, title: string): Promise<SessionDetail> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  return unwrap<SessionDetail>(response);
}

export async function deleteSession(sessionId: string): Promise<void> {
  // 删除会话时由后端一并清理该会话关联的历史消息和小结文件。
  const response = await fetch(`${API_BASE}/sessions/${sessionId}`, {
    method: "DELETE",
  });
  await unwrap<{ session_id: string; status: string }>(response);
}

export async function getSessionMessages(sessionId: string): Promise<SessionMessage[]> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/messages`);
  return unwrap<SessionMessage[]>(response);
}

export async function listSessionAgentRuns(sessionId: string): Promise<AgentTaskRunSummary[]> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/agent/runs`);
  return unwrap<AgentTaskRunSummary[]>(response);
}

export async function getAgentRunDetail(runId: string): Promise<AgentRunDetail> {
  const response = await fetch(`${API_BASE}/agent/runs/${runId}`);
  return unwrap<AgentRunDetail>(response);
}

export async function getKnowledgeSources(): Promise<KnowledgeSourcesResponse> {
  const response = await fetch(`${API_BASE}/knowledge/sources`);
  return unwrap<KnowledgeSourcesResponse>(response);
}
