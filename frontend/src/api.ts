import type {
  AgentResponse,
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

export async function uploadReport(file: File): Promise<ReportParseResult> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/reports/upload`, {
    method: "POST",
    body: formData,
  });
  return unwrap<ReportParseResult>(response);
}

export async function sendChat(payload: {
  session_id?: string | null;
  report_id?: string | null;
  message: string;
}): Promise<AgentResponse> {
  const response = await fetch(`${API_BASE}/agent/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return unwrap<AgentResponse>(response);
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

export async function bootstrapKnowledge(): Promise<{ ingested: number; skipped: number; message: string }> {
  const response = await fetch(`${API_BASE}/knowledge/bootstrap`, {
    method: "POST",
  });
  return unwrap<{ ingested: number; skipped: number; message: string }>(response);
}

export async function getKnowledgeSources(): Promise<KnowledgeSourcesResponse> {
  const response = await fetch(`${API_BASE}/knowledge/sources`);
  return unwrap<KnowledgeSourcesResponse>(response);
}
