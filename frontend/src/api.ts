import type {
  ChatStreamEvent,
  KnowledgeSourcesResponse,
  ReportParseResult,
  SummaryArtifact,
} from "./types";

const API_BASE = "http://localhost:8000/api";

async function unwrap<T>(response: Response): Promise<T> {
  // 普通 JSON 接口统一在这里做错误处理，避免每个 API 都重复写样板代码。
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "请求失败。" }));
    throw new Error(error.detail ?? "请求失败。");
  }
  return (await response.json()) as T;
}

function parseSseBlock(rawEvent: string): ChatStreamEvent | null {
  // 后端返回的是标准 SSE 文本块，这里负责把一段原始文本拆成
  // `{ event, data }` 这种前端更容易消费的对象。
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
  // 上传接口使用 multipart/form-data，因为浏览器文件上传最自然的方式就是 FormData。
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
  // 这里没有使用 EventSource，是因为聊天接口是 POST，
  // 而原生 EventSource 只支持 GET。于是改用 fetch + ReadableStream 手动解析 SSE。
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
    // SSE 事件之间以空行分隔，因此这里不断查找 "\n\n" 边界。
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
  // 报告进度流是 GET，所以这里可以直接使用浏览器原生 EventSource。
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
  // 小结生成是普通 POST，不需要流式。
  const response = await fetch(`${API_BASE}/summaries/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return unwrap<SummaryArtifact>(response);
}

export async function getKnowledgeSources(): Promise<KnowledgeSourcesResponse> {
  // 当前主页面不再直接展示知识库区块，但保留这个接口方便调试和扩展。
  const response = await fetch(`${API_BASE}/knowledge/sources`);
  return unwrap<KnowledgeSourcesResponse>(response);
}
