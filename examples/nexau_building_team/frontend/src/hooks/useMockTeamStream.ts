import { useCallback, useRef, useState } from "react";
import type { AgentState, TaskInfo } from "../types";
import { applyEnvelope } from "../utils/applyEnvelope";
import { Phase, LEADER_ID, TEAM_ID, RUN_ID, RFC_WRITER_ID } from "../mock/mockConstants";
import { getPhaseEvents, type TimedEnvelope } from "../mock/mockEvents";
import { MOCK_TASKS } from "../mock/mockTasks";

/**
 * Mock 版本的 useTeamStream，用于演示视频。
 *
 * 与 useTeamStream 接口一致，额外暴露 mockTasks 供 App 读取。
 * 通过 setTimeout 链逐个回放预编排事件，ask_user 处暂停等待用户交互。
 */
export function useMockTeamStream() {
  const [agents, setAgents] = useState<Record<string, AgentState>>({});
  const [isStreaming, setIsStreaming] = useState(false);
  const [error] = useState<string | null>(null);
  const [mockTasks, setMockTasks] = useState<TaskInfo[]>([]);

  const phaseRef = useRef<Phase>(Phase.REQUIREMENTS);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const resumeCallbackRef = useRef<((userMessage: string) => void) | null>(null);

  // 使用 ref 存储函数引用，打破 useCallback 循环依赖
  const advancePhaseRef = useRef<(nextPhase: Phase) => void>(() => {});
  const handlePhaseCompleteRef = useRef<(completedPhase: Phase) => void>(() => {});

  // ── 清除定时器 ──────────────────────────────────────────────────
  const cancelReplay = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // ── 回放引擎：逐个 emit 事件 ────────────────────────────────────
  const replaySequence = useCallback(
    (events: TimedEnvelope[], startIndex: number, onDone: () => void) => {
      const next = (idx: number) => {
        if (idx >= events.length) {
          onDone();
          return;
        }

        const { envelope, delay, pauseAfter, taskSnapshot } = events[idx];

        timerRef.current = setTimeout(() => {
          // 1. 应用事件到 agent 状态
          setAgents((prev) => applyEnvelope(prev, envelope));

          // 2. 如果有 task 快照，更新 tasks
          if (taskSnapshot) {
            setMockTasks(taskSnapshot);
          }

          // 3. 如果需要暂停（ask_user），通知 phase 完成以设置 resume callback
          if (pauseAfter) {
            onDone();
            return;
          }

          // 4. 继续下一个事件
          next(idx + 1);
        }, delay);
      };

      next(startIndex);
    },
    [],
  );

  // ── Phase 推进逻辑 ──────────────────────────────────────────────
  advancePhaseRef.current = (nextPhase: Phase) => {
    phaseRef.current = nextPhase;
    setMockTasks(MOCK_TASKS[nextPhase] ?? []);

    const events = getPhaseEvents(nextPhase);
    if (events.length === 0) {
      setIsStreaming(false);
      return;
    }

    replaySequence(events, 0, () => {
      handlePhaseCompleteRef.current?.(nextPhase);
    });
  };

  // ── Phase 完成后的逻辑 ────────────────────────────────────────
  handlePhaseCompleteRef.current = (completedPhase: Phase) => {
    switch (completedPhase) {
      case Phase.REQUIREMENTS:
        phaseRef.current = Phase.ASK_USER_1;
        resumeCallbackRef.current = (userAnswer: string) => {
          setAgents((prev) =>
            applyEnvelope(prev, {
              team_id: TEAM_ID,
              agent_id: LEADER_ID,
              role_name: "user",
              run_id: RUN_ID,
              event: { type: "USER_MESSAGE", content: userAnswer },
            }),
          );
          advancePhaseRef.current?.(Phase.DESIGN);
        };
        break;

      case Phase.DESIGN:
        phaseRef.current = Phase.ASK_USER_2;
        resumeCallbackRef.current = (userAnswer: string) => {
          setAgents((prev) =>
            applyEnvelope(prev, {
              team_id: TEAM_ID,
              agent_id: RFC_WRITER_ID,
              role_name: "user",
              run_id: RUN_ID,
              event: { type: "USER_MESSAGE", content: userAnswer },
            }),
          );
          advancePhaseRef.current?.(Phase.IMPLEMENTATION);
        };
        break;

      case Phase.IMPLEMENTATION:
        phaseRef.current = Phase.COMPLETE;
        setMockTasks(MOCK_TASKS[Phase.COMPLETE] ?? []);
        setIsStreaming(false);
        break;

      default:
        setIsStreaming(false);
        break;
    }
  };

  // ═══════════════════════════════════════════════════════════════════
  //  公开接口（与 useTeamStream 一致）
  // ═══════════════════════════════════════════════════════════════════

  const startStream = useCallback(
    async (_message: string, _userId: string, _sessionId: string) => {
      setAgents({});
      setIsStreaming(true);
      setMockTasks(MOCK_TASKS[Phase.REQUIREMENTS] ?? []);
      phaseRef.current = Phase.REQUIREMENTS;
      resumeCallbackRef.current = null;

      const events = getPhaseEvents(Phase.REQUIREMENTS);
      replaySequence(events, 0, () => {
        handlePhaseCompleteRef.current?.(Phase.REQUIREMENTS);
      });
    },
    [replaySequence],
  );

  const sendUserMessage = useCallback(
    async (content: string, _toAgentId: string = "leader") => {
      const callback = resumeCallbackRef.current;
      if (!callback) return;
      resumeCallbackRef.current = null;
      callback(content);
    },
    [],
  );

  const stopStream = useCallback(() => {
    cancelReplay();
    setIsStreaming(false);
    resumeCallbackRef.current = null;
  }, [cancelReplay]);

  const loadHistory = useCallback(
    async (_userId: string, _sessionId: string): Promise<number> => 0,
    [],
  );

  const subscribeToStream = useCallback(
    async (_userId: string, _sessionId: string, _after: number) => {},
    [],
  );

  return {
    agents,
    setAgents,
    isStreaming,
    error,
    startStream,
    sendUserMessage,
    stopStream,
    loadHistory,
    subscribeToStream,
    mockTasks,
  };
}
