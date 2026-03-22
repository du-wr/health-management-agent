import type {
  ChatStreamEvent,
  KnowledgeSourcesResponse,
  ReportParseResult,
  SummaryArtifact,
} from "./types";

const API_BASE = "http://localhost:8000/api";

async function unwrap<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "请求失败。" }));
    throw new Error(error.detail ?? "请求失败。");
  }
  return (await response.json()) as T;
}

function parseSseBlock(rawEvent: string): ChatStreamEvent | null {
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
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/reports/upload`, {
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

export function streamReportProgress(
  reportId: string,
  onEvent: (event: ChatStreamEvent) => void,
): () => void {
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

export async function getKnowledgeSources(): Promise<KnowledgeSourcesResponse> {
  const response = await fetch(`${API_BASE}/knowledge/sources`);
  return unwrap<KnowledgeSourcesResponse>(response);
}
