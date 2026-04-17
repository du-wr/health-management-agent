export type LabStatus = "high" | "low" | "normal" | "unknown";
export type IntentName =
  | "report_follow_up"
  | "term_explanation"
  | "symptom_rag_advice"
  | "collect_more_info"
  | "safety_handoff";
export type TrustTier = "A" | "B" | "C";

export interface LabItem {
  name: string;
  value_raw: string;
  value_num?: number | null;
  unit: string;
  reference_range: string;
  status: LabStatus;
  clinical_note?: string | null;
}

export interface ReportParseResult {
  report_id: string;
  file_name: string;
  items: LabItem[];
  abnormal_items: LabItem[];
  raw_text: string;
  parse_warnings: string[];
  parse_status: string;
}

export interface ReportProgressPayload {
  report_id: string;
  stage: string;
  label: string;
  progress: number;
  parse_status: string;
  done: boolean;
  error?: string | null;
}

export interface Citation {
  source_type: string;
  doc_id: string;
  title: string;
  url: string;
  trust_tier: TrustTier;
  snippet: string;
}

export interface AgentDebug {
  analysis?: Record<string, unknown>;
  plan?: Record<string, unknown>;
  synthesis?: Record<string, unknown>;
  replan?: Record<string, unknown>;
  memory?: Record<string, unknown>;
  goal?: Record<string, unknown>;
  task_run?: Record<string, unknown>;
  trace_summary?: Array<Record<string, unknown>>;
}

export interface AgentResponse {
  session_id: string;
  intent: IntentName;
  answer: string;
  citations: Citation[];
  used_tools: string[];
  follow_up_questions: string[];
  safety_level: "safe" | "caution" | "handoff";
  handoff_required: boolean;
  debug?: AgentDebug | null;
}

export interface SummaryArtifact {
  summary_id: string;
  markdown: string;
  pdf_path: string;
  created_at: string;
}

export interface SessionReportInfo {
  report_id: string;
  file_name: string;
  parse_status: string;
}

export interface SessionSummary {
  session_id: string;
  title: string;
  created_at: string;
  last_message_at: string;
  message_count: number;
  last_message_preview: string;
  report?: SessionReportInfo | null;
}

export interface SessionDetail {
  session_id: string;
  title: string;
  created_at: string;
  last_message_at: string;
  message_count: number;
  report?: SessionReportInfo | null;
}

export interface SessionMessage {
  message_id: string;
  agent_run_id?: string | null;
  role: string;
  content: string;
  intent?: string | null;
  safety_level: "safe" | "caution" | "handoff";
  citations: Citation[];
  created_at: string;
}

export interface KnowledgeDoc {
  doc_id: string;
  title: string;
  url: string;
  source_domain: string;
  source_org: string;
  trust_tier: TrustTier;
  content_type: string;
  published_at?: string | null;
  snippet: string;
}

export interface KnowledgeSourcesResponse {
  total_docs: number;
  trust_breakdown: Record<string, number>;
  recent_docs: KnowledgeDoc[];
}

export interface ChatStreamEvent {
  event: string;
  data: unknown;
}
