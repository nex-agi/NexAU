import type { AgentState, TeamStreamEnvelope } from "../types";

/**
 * Pure reducer: apply a single TeamStreamEnvelope to the agents state.
 *
 * Used by both live SSE streaming and history replay so the
 * state-building logic is shared in one place.
 */
export function applyEnvelope(
  prev: Record<string, AgentState>,
  envelope: TeamStreamEnvelope
): Record<string, AgentState> {
  const { agent_id, role_name, event } = envelope;

  const existing = prev[agent_id] ?? {
    agentId: agent_id,
    roleName: role_name ?? agent_id,
    blocks: [],
    isActive: true,
  };
  const blocks = [...existing.blocks];
  const updated: AgentState = {
    ...existing,
    blocks,
    isActive: true,
  };

  if (event.type === "THINKING_TEXT_MESSAGE_CONTENT" && event.delta) {
    const last = blocks[blocks.length - 1];
    if (last && last.kind === "thinking") {
      blocks[blocks.length - 1] = {
        ...last,
        content: last.content + event.delta,
      };
    } else {
      blocks.push({ kind: "thinking", content: event.delta });
    }
  } else if (event.type === "THINKING_TEXT_MESSAGE_START") {
    // Push a new thinking block if the last block isn't already a thinking block
    const last = blocks[blocks.length - 1];
    if (!last || last.kind !== "thinking") {
      blocks.push({ kind: "thinking", content: "" });
    }
  } else if (event.type === "THINKING_TEXT_MESSAGE_END") {
    // No-op — thinking block is already accumulated
  } else if (event.type === "TEXT_MESSAGE_CONTENT" && event.delta) {
    const last = blocks[blocks.length - 1];
    if (last && last.kind === "text") {
      blocks[blocks.length - 1] = {
        ...last,
        content: last.content + event.delta,
      };
    } else {
      blocks.push({ kind: "text", content: event.delta });
    }
  } else if (event.type === "TOOL_CALL_START") {
    blocks.push({
      kind: "tool_call",
      id: event.tool_call_id ?? "",
      name: event.tool_call_name ?? event.name ?? "",
      args: "",
      argsDone: false,
      result: null,
    });
  } else if (event.type === "TOOL_CALL_ARGS" && event.delta) {
    const toolCallId = event.tool_call_id ?? "";
    let updatedAny = false;

    for (let i = blocks.length - 1; i >= 0; i--) {
      const b = blocks[i];
      if (
        b.kind === "tool_call" &&
        (toolCallId ? b.id === toolCallId : true)
      ) {
        blocks[i] = { ...b, args: b.args + event.delta };
        updatedAny = true;
        break;
      }
    }

    if (!updatedAny) {
      blocks.push({
        kind: "tool_call",
        id: toolCallId,
        name: "",
        args: event.delta,
        argsDone: false,
        result: null,
      });
    }
  } else if (event.type === "TOOL_CALL_END") {
    const toolCallId = event.tool_call_id ?? "";

    for (let i = blocks.length - 1; i >= 0; i--) {
      const b = blocks[i];
      if (
        b.kind === "tool_call" &&
        (toolCallId ? b.id === toolCallId : true)
      ) {
        blocks[i] = { ...b, argsDone: true };
        break;
      }
    }
  } else if (event.type === "TOOL_CALL_RESULT") {
    const toolCallId = event.tool_call_id ?? "";
    const resultContent = event.content ?? "";
    let updatedAny = false;

    if (toolCallId) {
      for (let i = blocks.length - 1; i >= 0; i--) {
        const b = blocks[i];
        if (b.kind === "tool_call" && b.id === toolCallId) {
          blocks[i] = { ...b, argsDone: true, result: resultContent };
          updatedAny = true;
          break;
        }
      }
    }

    if (!updatedAny) {
      for (let i = blocks.length - 1; i >= 0; i--) {
        const b = blocks[i];
        if (b.kind === "tool_call" && b.result === null) {
          blocks[i] = { ...b, argsDone: true, result: resultContent };
          break;
        }
      }
    }
  } else if (event.type === "RUN_FINISHED") {
    updated.isActive = false;
  } else if (event.type === "RUN_ERROR" && event.message) {
    blocks.push({ kind: "error", message: event.message });
    updated.isActive = false;
  } else if (event.type === "USER_MESSAGE" && event.content) {
    blocks.push({ kind: "user_message", content: event.content });
  } else if (event.type === "TEAM_MESSAGE" && event.content) {
    blocks.push({
      kind: "team_message",
      from: event.from_agent_id ?? "unknown",
      content: event.content,
    });
  }

  return { ...prev, [agent_id]: updated };
}
