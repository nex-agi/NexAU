/** Matches Python TeamStreamEnvelope. */
export interface TeamStreamEnvelope {
  team_id: string;
  agent_id: string;
  role_name: string | null;
  run_id: string | null;
  event: AgentEvent;
}

/** SSE wrapper from /team/stream. */
export interface TeamStreamEnvelopeResponse {
  type: "team_event" | "complete" | "error";
  envelope: TeamStreamEnvelope | null;
  session_id: string;
  error: string | null;
}

/** Subset of Event types we care about for rendering. */
export interface AgentEvent {
  type: string;
  delta?: string;
  message_id?: string;
  run_id?: string;
  agent_id?: string;
  name?: string;
  tool_call_id?: string;
  tool_call_name?: string;
  // UserMessageEvent / TeamMessageEvent fields
  content?: string;
  to_agent_id?: string;
  from_agent_id?: string;
  // RunErrorEvent fields
  message?: string;
}

export interface ToolCallBlock {
  kind: "tool_call";
  id: string;
  name: string;
  /** Raw streamed JSON args (may be partial / invalid until TOOL_CALL_END). */
  args: string;
  /** True after TOOL_CALL_END is observed for this tool_call_id. */
  argsDone: boolean;
  /** Raw TOOL_CALL_RESULT content (JSON string from backend). */
  result: string | null;
}

/** A content block — either text, thinking, a tool call, a message, or an error, rendered in chronological order. */
export type ContentBlock =
  | { kind: "text"; content: string }
  | { kind: "thinking"; content: string }
  | ToolCallBlock
  | { kind: "user_message"; content: string }
  | { kind: "team_message"; from: string; content: string }
  | { kind: "error"; message: string };

/** Per-agent accumulated state with interleaved content blocks. */
export interface AgentState {
  agentId: string;
  roleName: string;
  blocks: ContentBlock[];
  isActive: boolean;
}

/** Task from /team/tasks. */
export interface TaskInfo {
  task_id: string;
  title: string;
  description: string;
  status: string;
  priority: number;
  dependencies: string[];
  assignee_agent_id: string | null;
  result_summary: string | null;
  created_by: string;
  is_blocked: boolean;
}
