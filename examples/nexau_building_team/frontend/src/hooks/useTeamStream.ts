import { useCallback, useRef, useState } from "react";
import type { AgentState, TeamStreamEnvelope, TeamStreamEnvelopeResponse } from "../types";
import { applyEnvelope } from "../utils/applyEnvelope";

/** Hook for connecting to /team/stream SSE endpoint with history support. */
export function useTeamStream() {
  const [agents, setAgents] = useState<Record<string, AgentState>>({});
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const activeSessionRef = useRef<{
    userId: string;
    sessionId: string;
  } | null>(null);

  /** Fetch and replay stored events from the backend. Returns the number of events loaded. */
  const loadHistory = useCallback(
    async (userId: string, sessionId: string): Promise<number> => {
      try {
        const url = `/team/history?user_id=${encodeURIComponent(userId)}&session_id=${encodeURIComponent(sessionId)}`;
        console.log("[loadHistory] fetching:", url);
        const res = await fetch(url);
        if (!res.ok) {
          console.warn("[loadHistory] HTTP error:", res.status, res.statusText);
          return 0;
        }
        const events: TeamStreamEnvelope[] = await res.json();
        console.log("[loadHistory] got", events.length, "events");
        if (!events.length) return 0;
        setAgents(() => {
          let state: Record<string, AgentState> = {};
          for (const envelope of events) {
            if (envelope.agent_id && envelope.event) {
              state = applyEnvelope(state, envelope);
            }
          }
          return state;
        });
        return events.length;
      } catch (err) {
        console.error("[loadHistory] error:", err);
        return 0;
      }
    },
    []
  );

  const startStream = useCallback(
    async (message: string, userId: string, sessionId: string) => {
      // 中断已有的 subscribe/stream 连接，避免重复接收事件
      abortRef.current?.abort();
      setIsStreaming(true);
      setError(null);
      // 不清空 agents — 保留上一轮的内容，新事件会追加到已有 blocks
      activeSessionRef.current = { userId, sessionId };

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const response = await fetch("/team/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: userId,
            session_id: sessionId,
            message,
          }),
          signal: controller.signal,
        });

        if (!response.ok || !response.body) {
          throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const json = line.slice(6).trim();
            if (!json) continue;

            try {
              const msg: TeamStreamEnvelopeResponse = JSON.parse(json);
              if (msg.type === "error") {
                setError(msg.error);
                return;
              }
              if (msg.type === "complete") {
                return;
              }
              if (msg.type === "team_event" && msg.envelope) {
                setAgents((prev) => applyEnvelope(prev, msg.envelope!));
              }
            } catch {
              // skip malformed JSON
            }
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          setError((err as Error).message);
        }
      } finally {
        setIsStreaming(false);
        activeSessionRef.current = null;
      }
    },
    []
  );

  const sendUserMessage = useCallback(
    async (content: string, toAgentId: string = "leader") => {
      const session = activeSessionRef.current;
      if (!session) return;

      try {
        await fetch("/team/user-message", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: session.userId,
            session_id: session.sessionId,
            content,
            to_agent_id: toAgentId,
          }),
        });
      } catch (err) {
        setError((err as Error).message);
      }
    },
    []
  );

  const stopStream = useCallback(() => {
    const session = activeSessionRef.current;
    if (session) {
      fetch("/team/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: session.userId,
          session_id: session.sessionId,
        }),
      }).catch(() => {});
    }
    abortRef.current?.abort();
    setIsStreaming(false);
    activeSessionRef.current = null;
  }, []);

  /** Reconnect to an in-progress team run after browser refresh.
   *  Subscribes to /team/subscribe starting from `after` (already-loaded event count). */
  const subscribeToStream = useCallback(
    async (userId: string, sessionId: string, after: number) => {
      // 如果 startStream 已在运行，跳过重连（避免重复接收事件）
      if (activeSessionRef.current) return;

      // Check if team is still running before subscribing
      try {
        const statusRes = await fetch(
          `/team/status?user_id=${encodeURIComponent(userId)}&session_id=${encodeURIComponent(sessionId)}`
        );
        if (!statusRes.ok) return;
        const status = await statusRes.json();
        if (!status.running) return;
      } catch {
        return;
      }

      // Re-check after async fetch — startStream may have started during the await
      if (activeSessionRef.current) return;

      setIsStreaming(true);
      setError(null);
      activeSessionRef.current = { userId, sessionId };

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const url = `/team/subscribe?user_id=${encodeURIComponent(userId)}&session_id=${encodeURIComponent(sessionId)}&after=${after}`;
        const response = await fetch(url, { signal: controller.signal });
        if (!response.ok || !response.body) {
          throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const json = line.slice(6).trim();
            if (!json) continue;
            try {
              const msg = JSON.parse(json);
              if (msg.type === "error") { setError(msg.error); return; }
              if (msg.type === "complete") return;
              if (msg.type === "team_event" && msg.envelope) {
                setAgents((prev) => applyEnvelope(prev, msg.envelope));
              }
            } catch { /* skip malformed */ }
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          setError((err as Error).message);
        }
      } finally {
        // Only clean up if we're still the active consumer
        // (startStream may have taken over by replacing abortRef.current)
        if (abortRef.current === controller) {
          setIsStreaming(false);
          activeSessionRef.current = null;
        }
      }
    },
    []
  );

  return { agents, setAgents, isStreaming, error, startStream, sendUserMessage, stopStream, loadHistory, subscribeToStream };
}
