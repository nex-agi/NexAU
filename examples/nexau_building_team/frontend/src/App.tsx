import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  User, Cpu, PenTool, Wrench, ShieldCheck,
  CheckCircle2, Square,
  MessageSquare, Zap,
  Send, X, Box, AlertTriangle, FolderOpen,
  Search, Brain, Pencil, MessageCircle, ListTodo,
  Bot, BotMessageSquare, CircuitBoard, Cog, Terminal, Blocks, Hammer,
} from "lucide-react";
import { useTeamStream } from "./hooks/useTeamStream";
import { useMockTeamStream } from "./hooks/useMockTeamStream";
import { AgentPanel } from "./components/AgentPanel";
import { FileTree } from "./components/FileTree";
import { FileViewer } from "./components/FileViewer";
import { parseAskUserQuestions } from "./components/AskUserBlock";
import { AskUserModal } from "./components/AskUserModal";
import { TaskDetailModal } from "./components/TaskDetailModal";
import { ArtifactPanel } from "./components/ArtifactPanel";
import { MOCK_FILE_TREE, MOCK_FILE_CONTENTS } from "./mock/mockFiles";
import type { AgentState, TaskInfo } from "./types";

const USER_ID = "demo-user";
const SESSION_KEY = "nexau-session-id";

function getOrCreateSession(): string {
  const s = localStorage.getItem(SESSION_KEY);
  if (s) return s;
  const id = `session-${Date.now()}`;
  localStorage.setItem(SESSION_KEY, id);
  return id;
}

type LucideIcon = typeof Cpu;

const BUILDER_COLOR = "#2563eb";
const BUILDER_ICONS: LucideIcon[] = [
  Wrench, Bot, BotMessageSquare, CircuitBoard, Cog, Terminal, Hammer, Blocks,
];

function hashCode(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function agentVisuals(
  role: string,
  isFirst: boolean,
  agentId?: string,
): { icon: LucideIcon; color: string } {
  const r = role.toLowerCase();
  if (isFirst || r.includes("leader") || r.includes("orchestrat"))
    return { icon: Cpu, color: "#6366f1" };
  if (r.includes("build") || r.includes("eng")) {
    const idx = agentId ? hashCode(agentId) % BUILDER_ICONS.length : 0;
    return { icon: BUILDER_ICONS[idx], color: BUILDER_COLOR };
  }
  if (r.includes("test") || r.includes("qa"))
    return { icon: ShieldCheck, color: "#ea580c" };
  if (r.includes("writ") || r.includes("doc") || r.includes("rfc"))
    return { icon: PenTool, color: "#0284c7" };
  return { icon: Box, color: "#2563eb" };
}

interface AgentActivity { label: string; icon: LucideIcon; color: string }

/** 根据 agent 最后一个 block 推断当前活动类别 */
function detectActivity(agent: AgentState): AgentActivity | null {
  if (agent.blocks.length === 0) return null;
  for (let i = agent.blocks.length - 1; i >= 0; i--) {
    const b = agent.blocks[i];
    if (b.kind === "thinking")
      return agent.isActive ? { label: "思考中", icon: Brain, color: "#8b5cf6" } : { label: "已完成", icon: CheckCircle2, color: "#10b981" };
    if (b.kind === "tool_call") {
      if (!agent.isActive) return { label: "已完成", icon: CheckCircle2, color: "#10b981" };
      const n = b.name.toLowerCase();
      if (n === "ask_user")
        return { label: "对话中", icon: MessageCircle, color: "#059669" };
      if (/read|search|grep|glob|find|list|get/.test(n))
        return { label: "调研中", icon: Search, color: "#0ea5e9" };
      if (/write|edit|create|append|delete|remove/.test(n))
        return { label: "编写中", icon: Pencil, color: "#f59e0b" };
      return { label: "调研中", icon: Search, color: "#0ea5e9" };
    }
    if (b.kind === "text")
      return agent.isActive ? { label: "对话中", icon: MessageCircle, color: "#059669" } : { label: "已完成", icon: CheckCircle2, color: "#10b981" };
    if (b.kind === "user_message" || b.kind === "team_message")
      return agent.isActive ? { label: "对话中", icon: MessageCircle, color: "#059669" } : null;
  }
  return null;
}

interface LayoutNode {
  id: string;
  role: string;
  name: string;
  x: number;
  y: number;
  isActive: boolean;
  color: string;
  icon: LucideIcon;
  isVirtual?: boolean;
}

interface StreamMessage {
  id: string;
  fromId: string;
  fromName: string;
  toId?: string;
  toName?: string;
  text: string;
  type: "user" | "team";
}

// ---------------------------------------------------------------------------
// Mock mode detection
// ---------------------------------------------------------------------------
const IS_MOCK_MODE = new URLSearchParams(window.location.search).get("mock") === "true";

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
export default function App() {
  const [sessionId, setSessionId] = useState(getOrCreateSession);
  const [sessions, setSessions] = useState<string[]>([]);

  // 两个 hook 都无条件调用（React 规则），根据 mock mode 选择使用哪个
  const realStream = useTeamStream();
  const mockStream = useMockTeamStream();
  const stream = IS_MOCK_MODE ? mockStream : realStream;

  const {
    agents, setAgents, isStreaming, error,
    startStream, sendUserMessage, stopStream,
    loadHistory, subscribeToStream,
  } = stream;

  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [selectedTask, setSelectedTask] = useState<TaskInfo | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [targetAgent, setTargetAgent] = useState("leader");
  const [rightTab, setRightTab] = useState<"traces" | "files">("traces");
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [submittedAskIds, setSubmittedAskIds] = useState<Set<string>>(
    new Set(),
  );
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const composingRef = useRef(false);

  // --- Drag support for agent nodes ---
  const topoRef = useRef<HTMLDivElement>(null);
  const [dragPositions, setDragPositions] = useState<Record<string, { x: number; y: number }>>({});
  const dragRef = useRef<{ nodeId: string; startX: number; startY: number; origX: number; origY: number } | null>(null);
  const wasDragging = useRef(false);

  // --- Session management ---
  const refreshSessions = useCallback(() => {
    if (IS_MOCK_MODE) return;
    fetch(`/team/sessions?user_id=${encodeURIComponent(USER_ID)}`)
      .then((r) => r.json())
      .then((list: string[]) => setSessions(list))
      .catch(() => {});
  }, []);

  useEffect(() => { refreshSessions(); }, [refreshSessions]);
  useEffect(() => { if (!isStreaming) refreshSessions(); }, [isStreaming, refreshSessions]);

  useEffect(() => {
    if (IS_MOCK_MODE) return;
    loadHistory(USER_ID, sessionId).then((count) => {
      subscribeToStream(USER_ID, sessionId, count);
    });
  }, [loadHistory, subscribeToStream, sessionId]);

  // --- Task polling (mock mode: read from hook state) ---
  const currentMockTasks = IS_MOCK_MODE && "mockTasks" in stream
    ? (stream as typeof mockStream).mockTasks
    : null;

  useEffect(() => {
    if (IS_MOCK_MODE) {
      if (currentMockTasks) {
        setTasks(currentMockTasks);
      }
      return;
    }
    const poll = () => {
      fetch(
        `/team/tasks?user_id=${encodeURIComponent(USER_ID)}&session_id=${encodeURIComponent(sessionId)}`,
      )
        .then((r) => r.json())
        .then((t: TaskInfo[]) => setTasks(t))
        .catch(() => {});
    };
    poll();
    if (!isStreaming) return;
    const iv = setInterval(poll, 3000);
    return () => clearInterval(iv);
  }, [isStreaming, sessionId, currentMockTasks]);

  // --- Auto-scroll messages ---
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [agents]);

  // --- Send handler ---
  const handleSend = useCallback(() => {
    const msg = chatInput.trim();
    if (!msg) return;
    setChatInput("");
    if (isStreaming) {
      sendUserMessage(msg, targetAgent);
    } else {
      startStream(msg, USER_ID, sessionId);
    }
  }, [chatInput, isStreaming, targetAgent, sendUserMessage, startStream, sessionId]);

  const switchSession = useCallback(
    (newId: string) => {
      if (newId === sessionId || isStreaming) return;
      localStorage.setItem(SESSION_KEY, newId);
      setAgents({});
      setTasks([]);
      setSelectedAgentId(null);
      setSessionId(newId);
    },
    [sessionId, isStreaming, setAgents],
  );

  const newSession = useCallback(() => {
    const id = `session-${Date.now()}`;
    localStorage.setItem(SESSION_KEY, id);
    setAgents({});
    setTasks([]);
    setSelectedAgentId(null);
    setSessionId(id);
  }, [setAgents]);

  // --- Computed: layout nodes ---
  const agentIds = useMemo(() => Object.keys(agents), [agents]);

  const layoutNodes: LayoutNode[] = useMemo(() => {
    const list = Object.values(agents);
    // 1. 通过 roleName 识别 leader（包含 "leader" 或 "orchestrat"）
    const leaderAgent = list.find((a) => {
      const r = a.roleName.toLowerCase();
      return r.includes("leader") || r.includes("orchestrat");
    });
    const firstId = leaderAgent ? leaderAgent.agentId : (list.length > 0 ? list.sort((a, b) => a.agentId.localeCompare(b.agentId))[0].agentId : null);
    const workers = list.filter((a) => a.agentId !== firstId);

    const nodes: LayoutNode[] = [];

    // Virtual user node
    nodes.push({
      id: "__user__", role: "User", name: "You",
      x: 50, y: 12, isActive: isStreaming, color: "#059669",
      icon: User, isVirtual: true,
    });

    // Leader
    if (firstId) {
      const leader = agents[firstId];
      const v = agentVisuals(leader.roleName, true, leader.agentId);
      nodes.push({
        id: leader.agentId, role: "Leader",
        name: leader.agentId, x: 50, y: 38,
        isActive: leader.isActive, color: v.color, icon: v.icon,
      });
    }

    // Workers
    workers.sort((a, b) => a.agentId.localeCompare(b.agentId));
    workers.forEach((w, i) => {
      const v = agentVisuals(w.roleName, false, w.agentId);
      const x =
        workers.length === 1
          ? 50
          : 15 + (70 / (workers.length - 1)) * i;
      nodes.push({
        id: w.agentId, role: w.roleName || "Worker",
        name: w.agentId, x, y: 75,
        isActive: w.isActive, color: v.color, icon: v.icon,
      });
    });

    return nodes;
  }, [agents, isStreaming]);

  // 获取节点的有效位置（拖动覆盖 > 计算布局）
  const getNodePos = useCallback(
    (node: LayoutNode) => dragPositions[node.id] ?? { x: node.x, y: node.y },
    [dragPositions],
  );

  // 当 agent 列表变化时清除过期的拖动位置
  useEffect(() => {
    setDragPositions((prev) => {
      const ids = new Set(layoutNodes.map((n) => n.id));
      const next: Record<string, { x: number; y: number }> = {};
      for (const [k, v] of Object.entries(prev)) {
        if (ids.has(k)) next[k] = v;
      }
      return Object.keys(next).length === Object.keys(prev).length ? prev : next;
    });
  }, [layoutNodes]);

  const handleDragStart = useCallback(
    (nodeId: string, e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const node = layoutNodes.find((n) => n.id === nodeId);
      if (!node) return;
      const pos = dragPositions[nodeId] ?? { x: node.x, y: node.y };
      wasDragging.current = false;
      dragRef.current = { nodeId, startX: e.clientX, startY: e.clientY, origX: pos.x, origY: pos.y };

      const handleMove = (ev: MouseEvent) => {
        const d = dragRef.current;
        const rect = topoRef.current?.getBoundingClientRect();
        if (!d || !rect) return;
        wasDragging.current = true;
        const dx = ((ev.clientX - d.startX) / rect.width) * 100;
        const dy = ((ev.clientY - d.startY) / rect.height) * 100;
        const nx = Math.max(2, Math.min(98, d.origX + dx));
        const ny = Math.max(2, Math.min(98, d.origY + dy));
        setDragPositions((prev) => ({ ...prev, [d.nodeId]: { x: nx, y: ny } }));
      };

      const handleUp = () => {
        dragRef.current = null;
        window.removeEventListener("mousemove", handleMove);
        window.removeEventListener("mouseup", handleUp);
      };

      window.addEventListener("mousemove", handleMove);
      window.addEventListener("mouseup", handleUp);
    },
    [layoutNodes, dragPositions],
  );

  // --- Computed: per-agent activity bubble ---
  const agentActivities = useMemo(() => {
    const map: Record<string, AgentActivity | null> = {};
    for (const agent of Object.values(agents)) {
      map[agent.agentId] = detectActivity(agent);
    }
    return map;
  }, [agents]);

  // --- Computed: per-agent task progress (from Todo tool calls in trace) ---
  const agentTaskProgress = useMemo(() => {
    const map: Record<string, { done: number; total: number }> = {};
    for (const agent of Object.values(agents)) {
      // 从后往前找最后一次 todo/TodoWrite 工具调用，取其 args 中的 todos 列表
      for (let i = agent.blocks.length - 1; i >= 0; i--) {
        const b = agent.blocks[i];
        if (b.kind !== "tool_call") continue;
        if (!/todo/i.test(b.name)) continue;
        try {
          const parsed = JSON.parse(b.args);
          const todos: { status?: string }[] = parsed.todos ?? parsed.items ?? [];
          if (todos.length === 0) continue;
          const done = todos.filter((t) => t.status === "completed").length;
          map[agent.agentId] = { done, total: todos.length };
        } catch { /* partial / invalid JSON, skip */ }
        break; // 只取最后一次
      }
    }
    return map;
  }, [agents]);

  // --- Computed: per-agent current todo label (first in_progress, then first pending) ---
  const agentTodoLabel = useMemo(() => {
    const map: Record<string, string | null> = {};
    for (const agent of Object.values(agents)) {
      let label: string | null = null;
      for (let i = agent.blocks.length - 1; i >= 0; i--) {
        const b = agent.blocks[i];
        if (b.kind !== "tool_call" || !/todo/i.test(b.name)) continue;
        try {
          const parsed = JSON.parse(b.args);
          const todos: { description?: string; status?: string }[] = parsed.todos ?? parsed.items ?? [];
          const inProgress = todos.find((t) => t.status === "in_progress");
          if (inProgress?.description) { label = inProgress.description; break; }
          const pending = todos.find((t) => t.status === "pending");
          if (pending?.description) { label = pending.description; break; }
        } catch { /* partial JSON, skip */ }
        break;
      }
      map[agent.agentId] = label;
    }
    return map;
  }, [agents]);

  // --- Computed: per-task modified files ---
  // 按时间顺序扫描 agent blocks，通过 team_message 中的 task_id 切分任务边界，
  // 将文件写入归属到当前正在执行的 task。
  const taskModifiedFiles = useMemo(() => {
    const taskIds = tasks.map((t) => t.task_id);
    if (taskIds.length === 0) return {} as Record<string, string[]>;

    // 按 agent 分组 tasks
    const agentTaskIds: Record<string, string[]> = {};
    for (const t of tasks) {
      if (!t.assignee_agent_id) continue;
      (agentTaskIds[t.assignee_agent_id] ??= []).push(t.task_id);
    }

    const result: Record<string, Set<string>> = {};

    for (const [agentId, myTaskIds] of Object.entries(agentTaskIds)) {
      const agent = agents[agentId];
      if (!agent) continue;

      // 1. 按出现顺序扫描 blocks，检测当前 task
      let currentTaskId: string | null = null;
      for (const b of agent.blocks) {
        // 检测 task 边界：team_message 内容包含某个 task_id
        if (b.kind === "team_message" || b.kind === "user_message") {
          const content = b.content.toUpperCase();
          for (const tid of myTaskIds) {
            if (content.includes(tid.toUpperCase())) {
              currentTaskId = tid;
              break;
            }
          }
        }

        // 如果还没检测到 task，但 agent 只有一个 task，直接归属
        const effectiveTask = currentTaskId ?? (myTaskIds.length === 1 ? myTaskIds[0] : null);

        // 收集文件写入
        if (effectiveTask && b.kind === "tool_call") {
          const n = b.name.toLowerCase();
          if (!/write|edit|create|append|save/.test(n)) continue;
          try {
            const parsed = JSON.parse(b.args);
            const fp: string | undefined = parsed.file_path ?? parsed.path ?? parsed.filename;
            if (fp) (result[effectiveTask] ??= new Set()).add(fp);
          } catch { /* partial JSON */ }
        }
      }
    }

    const map: Record<string, string[]> = {};
    for (const [tid, files] of Object.entries(result)) {
      map[tid] = [...files];
    }
    return map;
  }, [tasks, agents]);

  // --- Helpers for display names ---
  const leaderId = useMemo(() => {
    const list = Object.values(agents);
    const leaderAgent = list.find((a) => {
      const r = a.roleName.toLowerCase();
      return r.includes("leader") || r.includes("orchestrat");
    });
    if (leaderAgent) return leaderAgent.agentId;
    const sorted = list.sort((a, b) => a.agentId.localeCompare(b.agentId));
    return sorted.length > 0 ? sorted[0].agentId : null;
  }, [agents]);

  /** Resolve a human-readable display name for an agent.
   *  Leader always shows "Leader". Duplicated roles fall back to agentId. */
  const resolveAgentName = useCallback(
    (agentId: string) => {
      const a = agents[agentId];
      if (!a) return agentId;
      // First agent is always the leader
      if (agentId === leaderId) return "Leader";
      // If multiple agents share the same roleName, use agentId directly
      const sameRole = Object.values(agents).filter((x) => x.roleName === a.roleName);
      if (sameRole.length > 1) return agentId;
      return a.roleName || agentId;
    },
    [agents, leaderId],
  );

  // --- Computed: messages (user_message + team_message from all agents) ---
  const streamMessages: StreamMessage[] = useMemo(() => {
    const msgs: StreamMessage[] = [];
    for (const agent of Object.values(agents)) {
      for (let i = 0; i < agent.blocks.length; i++) {
        const b = agent.blocks[i];
        if (b.kind === "user_message") {
          // Determine sender: explicit from_agent_id > infer leader for sub-agents > human user
          let senderId = "__user__";
          let senderName = "You";
          let msgType: "user" | "team" = "user";

          if (b.from && agents[b.from]) {
            // Explicit sender agent
            senderId = b.from;
            senderName = resolveAgentName(b.from);
            msgType = "team";
          } else if (leaderId && agent.agentId !== leaderId) {
            // Sub-agent received user_message without from → sender is the leader
            senderId = leaderId;
            senderName = resolveAgentName(leaderId);
            msgType = "team";
          }

          msgs.push({
            id: `${agent.agentId}-um-${i}`,
            fromId: senderId, fromName: senderName,
            toId: agent.agentId, toName: resolveAgentName(agent.agentId),
            text: b.content, type: msgType,
          });
        } else if (b.kind === "team_message") {
          msgs.push({
            id: `${agent.agentId}-tm-${i}`,
            fromId: b.from, fromName: resolveAgentName(b.from),
            toId: agent.agentId, toName: resolveAgentName(agent.agentId),
            text: b.content, type: "team",
          });
        }
      }
    }
    return msgs;
  }, [agents, leaderId, resolveAgentName]);

  // --- Latched ask_user: survives backend state changes until user submits ---
  const [activeAsk, setActiveAsk] = useState<{
    agentId: string; roleName: string; blockId: string;
    questions: ReturnType<typeof parseAskUserQuestions> & object;
    hidden: boolean;
  } | null>(null);

  // Detect new pending ask_user blocks and latch them
  useEffect(() => {
    if (activeAsk) return;

    for (const agent of Object.values(agents)) {
      for (let i = 0; i < agent.blocks.length; i++) {
        const b = agent.blocks[i];
        if (b.kind === "tool_call" && b.name === "ask_user") {
          const blockId = b.id || `tc-${i}`;
          if (submittedAskIds.has(blockId)) continue;
          // 判断用户是否已作答：检查后续 blocks 中是否有 user_message
          // （不能用 b.result !== null，因为 ask_user 工具会立即返回 result，
          //  TOOL_CALL_RESULT 与 TOOL_CALL_END 在同一渲染批次到达，导致弹窗永远不弹出）
          let alreadyAnswered = false;
          for (let j = i + 1; j < agent.blocks.length; j++) {
            const next = agent.blocks[j];
            if (next.kind === "user_message") { alreadyAnswered = true; break; }
            if (next.kind === "tool_call" && next.name === "ask_user") break;
          }
          if (alreadyAnswered) continue;
          // 等 args 完整后再弹出，避免 partial-json 导致内容不全
          if (!b.argsDone) continue;
          const questions = parseAskUserQuestions(b.args, false);
          if (questions && questions.length > 0) {
            setActiveAsk({ agentId: agent.agentId, roleName: agent.roleName, blockId, questions, hidden: false });
            return;
          }
        }
      }
    }
  }, [agents, submittedAskIds, activeAsk]);

  const dismissAsk = useCallback(() => {
    if (!activeAsk) return;
    setSubmittedAskIds((prev) => new Set(prev).add(activeAsk.blockId));
    setActiveAsk(null);
  }, [activeAsk]);

  const hideAsk = useCallback(() => {
    setActiveAsk((prev) => prev ? { ...prev, hidden: true } : null);
  }, []);

  const showAsk = useCallback(() => {
    setActiveAsk((prev) => prev ? { ...prev, hidden: false } : null);
  }, []);

  // --- Selected agent ---
  const selectedAgent: AgentState | null =
    selectedAgentId ? agents[selectedAgentId] ?? null : null;

  // --- Render ---
  return (
    <div className="flex h-screen bg-[#f8fafc] text-slate-800 font-sans overflow-hidden">

      {/* --- Left: Task Board + Artifact (full height) --- */}
      <div className="w-72 border-r border-slate-200 bg-white flex flex-col shrink-0">
        <ArtifactPanel
          tasks={tasks}
          agentTaskProgress={agentTaskProgress}
          taskModifiedFiles={taskModifiedFiles}
          resolveAgentName={resolveAgentName}
          onSelectTask={setSelectedTask}
          onSelectFile={setSelectedFile}
        />
      </div>

      {/* --- Center column: Header + Topology + Bottom --- */}
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* ===== Header ===== */}
        <header className="h-14 bg-white border-b border-slate-200 flex items-center justify-between px-6 shrink-0 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 bg-slate-800 flex items-center justify-center rounded-sm">
              <Zap size={16} className="text-white" />
            </div>
            <h1 className="font-semibold text-slate-900 tracking-tight text-sm uppercase">
              NexAU <span className="text-slate-400 font-normal">| Swarm Control</span>
            </h1>
          </div>
          <div className="flex items-center gap-3">
            {sessions.length > 0 && (
              <select
                value={sessionId}
                onChange={(e) => switchSession(e.target.value)}
                disabled={isStreaming}
                className="text-[11px] border border-slate-200 bg-white px-2 py-1 text-slate-700 disabled:opacity-50"
              >
                {sessions.map((sid) => (
                  <option key={sid} value={sid}>{sid.slice(0, 24)}</option>
                ))}
                {!sessions.includes(sessionId) && (
                  <option value={sessionId}>{sessionId.slice(0, 24)}</option>
                )}
              </select>
            )}
            <button
              onClick={newSession}
              disabled={isStreaming}
              className="text-[11px] px-3 py-1.5 border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 disabled:opacity-40 uppercase tracking-wider font-medium"
            >
              New Session
            </button>
            {isStreaming && (
              <button
                onClick={stopStream}
                className="flex items-center gap-1.5 text-[11px] px-3 py-1.5 border border-red-300 bg-red-50 text-red-600 hover:bg-red-100 uppercase tracking-wider font-medium"
              >
                <Square size={10} fill="currentColor" /> Stop
              </button>
            )}
          </div>
        </header>

        {error && (
          <div className="px-6 py-2 bg-red-50 border-b border-red-200 text-red-600 text-xs">
            Error: {error}
          </div>
        )}

        {/* --- Center: Swarm Topology --- */}
        <div ref={topoRef} className="flex-1 relative bg-[#f8fafc]">
          {/* Grid background */}
          <div className="absolute inset-0 bg-[linear-gradient(rgba(0,0,0,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(0,0,0,0.04)_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none" />

          {/* SVG connections */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
            {layoutNodes.map((node) => {
              if (node.isVirtual) return null;
              const leader = layoutNodes.find((n) => n.role === "Leader");
              if (!leader || node.id === leader.id) return null;
              const lp = getNodePos(leader);
              const np = getNodePos(node);
              const active = leader.isActive || node.isActive;
              return (
                <path
                  key={`link-${node.id}`}
                  d={`M ${lp.x} ${lp.y} C ${lp.x} ${(lp.y + np.y) / 2}, ${np.x} ${(lp.y + np.y) / 2}, ${np.x} ${np.y}`}
                  fill="none"
                  stroke={active ? node.color : "#e2e8f0"}
                  strokeWidth={active ? "1.5" : "1"}
                  className={active ? "line-flow" : ""}
                  vectorEffect="non-scaling-stroke"
                />
              );
            })}
            {/* User → Leader line */}
            {(() => {
              const userNode = layoutNodes.find((n) => n.isVirtual);
              const leaderNode = layoutNodes.find((n) => n.role === "Leader");
              if (!userNode || !leaderNode) return null;
              const up = getNodePos(userNode);
              const lp = getNodePos(leaderNode);
              return (
                <path
                  d={`M ${up.x} ${up.y} L ${lp.x} ${lp.y}`}
                  fill="none"
                  stroke={isStreaming ? "#059669" : "#e2e8f0"}
                  strokeWidth={isStreaming ? "1.5" : "1"}
                  className={isStreaming ? "line-flow" : ""}
                  vectorEffect="non-scaling-stroke"
                />
              );
            })()}
          </svg>

          {/* Nodes */}
          {layoutNodes.map((node) => {
            const isSelected = selectedAgentId === node.id;
            const pos = getNodePos(node);
            return (
              <div
                key={node.id}
                className="absolute flex flex-col items-center node-spawn z-10 cursor-pointer"
                style={{ left: `${pos.x}%`, top: `${pos.y}%`, transform: "translate(-50%, -50%)" }}
                onClick={() => {
                  if (wasDragging.current) { wasDragging.current = false; return; }
                  if (!node.isVirtual) setSelectedAgentId(isSelected ? null : node.id);
                }}
              >
                <div
                  className="w-12 h-12 flex items-center justify-center relative bg-white shadow-sm transition-colors cursor-grab active:cursor-grabbing"
                  style={{
                    border: `1.5px solid ${node.isActive ? node.color : isSelected ? node.color : "#cbd5e1"}`,
                    color: node.isActive || isSelected ? node.color : "#64748b",
                  }}
                  onMouseDown={(e) => handleDragStart(node.id, e)}
                >
                  {node.isActive && <div className="pulse-ring" style={{ color: node.color }} />}
                  <node.icon size={20} strokeWidth={1.5} />
                  {/* Hidden ask_user badge */}
                  {activeAsk?.hidden && activeAsk.agentId === node.id && (
                    <button
                      onClick={(e) => { e.stopPropagation(); showAsk(); }}
                      className="absolute -top-1.5 -right-1.5 bg-amber-500 text-white p-0.5 border-2 border-white animate-pulse cursor-pointer z-20"
                      title="Pending question — click to open"
                    >
                      <AlertTriangle size={10} strokeWidth={3} />
                    </button>
                  )}
                  {!node.isActive && !node.isVirtual && agents[node.id]?.blocks.length > 0 && !(activeAsk?.hidden && activeAsk.agentId === node.id) && (
                    <div className="absolute -bottom-1 -right-1 bg-emerald-500 text-white p-0.5 border-2 border-white">
                      <CheckCircle2 size={10} strokeWidth={3} />
                    </div>
                  )}
                </div>
                <div className={`mt-2 text-center bg-white px-2 py-1 border shadow-sm ${isSelected ? "border-slate-400" : "border-slate-200"}`}>
                  <p className="text-[9px] font-bold uppercase tracking-widest" style={{ color: node.color }}>{node.role}</p>
                  <p className="text-[10px] font-mono text-slate-500 truncate max-w-[80px]">{node.name}</p>
                </div>
                {/* Task progress bar */}
                {!node.isVirtual && agentTaskProgress[node.id] && (() => {
                  const p = agentTaskProgress[node.id];
                  const pct = p.total > 0 ? (p.done / p.total) * 100 : 0;
                  return (
                    <div className="mt-1 flex flex-col items-center w-20">
                      <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${pct}%`, backgroundColor: pct === 100 ? "#10b981" : node.color }}
                        />
                      </div>
                      <span className="text-[9px] text-slate-500 mt-0.5 font-mono">{p.done}/{p.total}</span>
                    </div>
                  );
                })()}
                {/* Activity speech bubble — side for leader, below for workers */}
                {!node.isVirtual && (agentTodoLabel[node.id] || agentActivities[node.id]) && (() => {
                  const todoLabel = agentTodoLabel[node.id];
                  const act = agentActivities[node.id];
                  const BubbleIcon = todoLabel ? ListTodo : act!.icon;
                  const color = todoLabel ? "#4D96FF" : act!.color;
                  const label = todoLabel ?? act!.label;
                  const isLeader = node.role === "Leader";
                  return isLeader ? (
                    <div className="absolute left-full ml-2 top-0 flex items-start pointer-events-none animate-in fade-in w-max z-20">
                      <div
                        className="w-0 h-0 shrink-0 mt-3 border-t-[5px] border-t-transparent border-b-[5px] border-b-transparent border-r-[5px]"
                        style={{ borderRightColor: color + "30" }}
                      />
                      <div
                        className="flex items-start gap-1 px-2 py-1 rounded-lg text-[10px] font-medium max-w-[200px]"
                        style={{ backgroundColor: color + "15", color, border: `1px solid ${color}30` }}
                      >
                        <BubbleIcon size={10} strokeWidth={2} className="shrink-0 mt-0.5" />
                        <span className="line-clamp-3 break-words leading-tight">{label}</span>
                      </div>
                    </div>
                  ) : (
                    <div className="mt-1.5 flex flex-col items-center pointer-events-none animate-in fade-in">
                      <div
                        className="w-0 h-0 border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent border-b-[5px]"
                        style={{ borderBottomColor: color + "30" }}
                      />
                      <div
                        className="flex items-start gap-1 px-2 py-1 rounded-lg text-[10px] font-medium max-w-[100px]"
                        style={{ backgroundColor: color + "15", color, border: `1px solid ${color}30` }}
                      >
                        <BubbleIcon size={10} strokeWidth={2} className="shrink-0 mt-0.5" />
                        <span className="line-clamp-3 break-words leading-tight">{label}</span>
                      </div>
                    </div>
                  );
                })()}
              </div>
            );
          })}

          {/* Empty state */}
          {Object.keys(agents).length === 0 && !isStreaming && (
            <div className="absolute inset-0 flex items-center justify-center">
              <p className="text-sm text-slate-400">Send a message to start the agent team</p>
            </div>
          )}
        </div>

        {/* ===== Bottom: Message Stream + Chat Input ===== */}
        <div className="h-56 border-t border-slate-200 bg-white shrink-0 flex flex-col">
        <div className="h-8 border-b border-slate-200 flex items-center px-4 gap-2 bg-slate-50">
          <MessageSquare size={14} className="text-slate-500" />
          <span className="text-[10px] uppercase tracking-widest text-slate-600 font-bold">Tracking Stream</span>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
          {streamMessages.length === 0 && (
            <p className="text-xs text-slate-400 italic text-center mt-4">No messages yet</p>
          )}
          {streamMessages.map((msg) => {
            const bgClass = msg.type === "user"
              ? "bg-emerald-50 border-emerald-200 text-emerald-800"
              : "bg-slate-50 border-slate-200 text-slate-700";
            return (
              <div key={msg.id} className={`p-2 px-3 text-sm border shadow-sm ${bgClass}`}>
                <span className="font-bold text-[11px] uppercase tracking-wider mr-2" style={{ color: msg.type === "user" ? "#059669" : "#6366f1" }}>
                  {msg.fromName}
                </span>
                {msg.toName && (
                  <span className="text-[10px] text-slate-400 mr-2">
                    ▶ <span className="text-blue-500">@{msg.toName}</span>
                  </span>
                )}
                <span className="text-[13px]">{msg.text}</span>
              </div>
            );
          })}
          <div ref={messagesEndRef} className="h-1" />
        </div>

        {/* Chat input */}
        <div className="h-12 border-t border-slate-200 flex items-center px-4 gap-2 bg-white shrink-0">
          {isStreaming && agentIds.length > 0 && (
            <select
              value={targetAgent}
              onChange={(e) => setTargetAgent(e.target.value)}
              className="text-[11px] border border-slate-200 bg-white px-2 py-1 text-slate-600"
            >
              {agentIds.map((id) => (
                <option key={id} value={id}>{id}</option>
              ))}
            </select>
          )}
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !composingRef.current) { e.preventDefault(); handleSend(); } }}
            onCompositionStart={() => { composingRef.current = true; }}
            onCompositionEnd={() => { composingRef.current = false; }}
            placeholder={isStreaming ? "Send message to agent..." : "Type a message to start the team..."}
            className="flex-1 text-sm px-3 py-2 border border-slate-200 bg-slate-50 outline-none focus:border-slate-400 placeholder:text-slate-400"
          />
          <button
            onClick={handleSend}
            disabled={!chatInput.trim()}
            className="p-2 text-slate-500 hover:text-slate-700 disabled:opacity-30"
          >
            <Send size={16} />
          </button>
        </div>
        </div>
      </div>

      {/* --- Right: Tabbed Panel (Traces + Files, full height) --- */}
      <div className="w-[380px] border-l border-slate-200 bg-white flex flex-col shrink-0">
        {/* Tab bar */}
        <div className="flex border-b border-slate-200 bg-slate-50 shrink-0">
          <button
            onClick={() => setRightTab("traces")}
            className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2.5 text-[11px] uppercase tracking-wider font-semibold transition-colors ${
              rightTab === "traces"
                ? "text-slate-800 border-b-2 border-slate-800"
                : "text-slate-400 hover:text-slate-600"
            }`}
          >
            <MessageSquare size={13} />
            Traces
          </button>
          <button
            onClick={() => setRightTab("files")}
            className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2.5 text-[11px] uppercase tracking-wider font-semibold transition-colors ${
              rightTab === "files"
                ? "text-slate-800 border-b-2 border-slate-800"
                : "text-slate-400 hover:text-slate-600"
            }`}
          >
            <FolderOpen size={13} />
            Files
          </button>
        </div>

        {/* Tab content */}
        {rightTab === "traces" && (
          <>
            {selectedAgent ? (
              <>
                <div className="p-3 border-b border-slate-200 flex items-center justify-between bg-slate-50">
                  <div className="flex items-center gap-2">
                    <MessageSquare size={14} className="text-slate-500" />
                    <h2 className="font-semibold text-xs tracking-wider uppercase text-slate-700">
                      {resolveAgentName(selectedAgent.agentId)}
                    </h2>
                    {selectedAgent.isActive && (
                      <span className="text-[9px] text-blue-500 font-bold animate-pulse uppercase">Active</span>
                    )}
                  </div>
                  <button onClick={() => setSelectedAgentId(null)} className="text-slate-400 hover:text-slate-600">
                    <X size={16} />
                  </button>
                </div>
                <div className="flex-1 overflow-y-auto p-4">
                  <AgentPanel key={selectedAgentId} agent={selectedAgent} onSendMessage={sendUserMessage} />
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center">
                <p className="text-xs text-slate-400 italic">Select an agent to view traces</p>
              </div>
            )}
          </>
        )}

        {rightTab === "files" && (
          <div className="flex-1 overflow-hidden">
              <FileTree
                selectedPath={selectedFile}
                onSelectFile={setSelectedFile}
                isStreaming={isStreaming}
                mockNodes={IS_MOCK_MODE ? MOCK_FILE_TREE : undefined}
              />
          </div>
        )}
      </div>

      {/* ===== ASK USER Modal ===== */}
      {activeAsk && !activeAsk.hidden && (
        <AskUserModal
          questions={activeAsk.questions}
          agentId={activeAsk.agentId}
          roleName={activeAsk.roleName}
          onSendMessage={sendUserMessage}
          onSubmitted={dismissAsk}
          onHide={hideAsk}
        />
      )}

      {/* ===== Task Detail Modal ===== */}
      {selectedTask && (
        <TaskDetailModal
          task={selectedTask}
          resolveAgentName={resolveAgentName}
          onClose={() => setSelectedTask(null)}
        />
      )}

      {/* ===== File Preview Modal ===== */}
      {selectedFile && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="bg-white w-[70vw] h-[80vh] flex flex-col shadow-xl rounded-lg overflow-hidden">
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-slate-200 bg-slate-50 shrink-0">
              <span className="text-xs font-mono text-slate-700 truncate">{selectedFile}</span>
              <button onClick={() => setSelectedFile(null)} className="text-slate-400 hover:text-slate-600">
                <X size={16} />
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <FileViewer filePath={selectedFile} mockFiles={IS_MOCK_MODE ? MOCK_FILE_CONTENTS : undefined} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
