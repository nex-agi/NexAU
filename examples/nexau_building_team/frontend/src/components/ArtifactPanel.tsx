import {
  FileJson, Terminal, Puzzle, Wrench, Rocket,
  ShieldCheck, Loader2, CheckCircle2, Circle, FolderOpen,
} from "lucide-react";
import type { TaskInfo } from "../types";

/** Task ID prefix → artifact category. */
const CATEGORIES = [
  { prefix: "REQ-", label: "需求文档", icon: Terminal, color: "#8b5cf6", bg: "bg-purple-50", border: "border-purple-500", text: "text-purple-600" },
  { prefix: "RFC-", label: "设计文档", icon: FileJson, color: "#0284c7", bg: "bg-sky-50", border: "border-sky-500", text: "text-sky-600" },
  { prefix: "CONFIG-", label: "智能体配置", icon: FileJson, color: "#6366f1", bg: "bg-indigo-50", border: "border-indigo-500", text: "text-indigo-600" },
  { prefix: "SKILL-", label: "Skills", icon: Puzzle, color: "#d97706", bg: "bg-amber-50", border: "border-amber-500", text: "text-amber-600" },
  { prefix: "BUILD-", label: "智能体脚本", icon: Rocket, color: "#2563eb", bg: "bg-blue-50", border: "border-blue-500", text: "text-blue-600" },
  { prefix: "TEST-", label: "测试验收", icon: ShieldCheck, color: "#059669", bg: "bg-emerald-50", border: "border-emerald-500", text: "text-emerald-600" },
] as const;

interface Props {
  tasks: TaskInfo[];
  agentTaskProgress: Record<string, { done: number; total: number }>;
  taskModifiedFiles: Record<string, string[]>;
  resolveAgentName: (id: string) => string;
  onSelectTask: (task: TaskInfo) => void;
  onSelectFile: (path: string) => void;
}

function statusIcon(status: string) {
  if (status === "completed") return <CheckCircle2 size={12} className="text-emerald-500 shrink-0" />;
  if (status === "in_progress") return <Loader2 size={12} className="text-blue-500 animate-spin shrink-0" />;
  return <Circle size={12} className="text-slate-300 shrink-0" />;
}

/** Compute per-task progress from assignee's TODO list. */
function taskProgress(
  task: TaskInfo,
  agentTaskProgress: Record<string, { done: number; total: number }>,
): { done: number; total: number; pct: number } | null {
  if (!task.assignee_agent_id) return null;
  const p = agentTaskProgress[task.assignee_agent_id];
  if (!p || p.total === 0) return null;
  return { ...p, pct: Math.round((p.done / p.total) * 100) };
}

export function ArtifactPanel({ tasks, agentTaskProgress, taskModifiedFiles, resolveAgentName, onSelectTask, onSelectFile }: Props) {
  // Group tasks by category prefix
  const grouped = CATEGORIES.map((cat) => {
    const items = tasks.filter((t) => t.task_id.startsWith(cat.prefix));
    return { ...cat, items };
  });

  // Tasks that don't match any known prefix
  const knownPrefixes = CATEGORIES.map((c) => c.prefix);
  const uncategorized = tasks.filter((t) => !knownPrefixes.some((p) => t.task_id.startsWith(p)));

  // Count empty categories as placeholder items for progress
  const emptyCategories = grouped.filter((cat) => cat.items.length === 0).length;
  const totalTasks = tasks.length + emptyCategories;
  const completedTasks = tasks.filter((t) => t.status === "completed").length;
  const overallPct = totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0;

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {/* Overall progress */}
      <div className="bg-white border border-slate-200 p-3 shadow-sm">
        <div className="flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-2">
          <span>构建进度</span>
          <span>{overallPct}%</span>
        </div>
        <div className="h-1.5 w-full bg-slate-100 overflow-hidden">
          <div
            className="h-full bg-emerald-500 transition-all duration-700"
            style={{ width: `${overallPct}%` }}
          />
        </div>
        <div className="mt-1.5 text-[9px] font-mono text-slate-400">
          {completedTasks} / {totalTasks} tasks completed
        </div>
      </div>

      {/* Category sections */}
      {grouped.map((cat) => {
        const CatIcon = cat.icon;
        const hasItems = cat.items.length > 0;
        const catTotal = hasItems ? cat.items.length : 1; // placeholder counts as 1
        const catDone = cat.items.filter((t) => t.status === "completed").length;
        const catPct = Math.round((catDone / catTotal) * 100);

        return (
          <div key={cat.prefix} className="space-y-2">
            {/* Category header */}
            <div className="flex items-center gap-2">
              <CatIcon size={14} style={{ color: cat.color }} />
              <span className={`text-[10px] font-bold uppercase tracking-widest ${cat.text}`}>
                {cat.label}
              </span>
              <span className="text-[9px] font-mono text-slate-400 ml-auto">
                {catDone}/{catTotal}
              </span>
            </div>

            {/* Category progress bar */}
            <div className="h-1 w-full bg-slate-100 overflow-hidden">
              <div
                className="h-full transition-all duration-500"
                style={{ width: `${catPct}%`, backgroundColor: cat.color }}
              />
            </div>

            {/* Placeholder card when no real tasks */}
            {!hasItems && (
              <div className="border border-dashed border-slate-200 p-2.5 bg-slate-50/50">
                <div className="flex items-center gap-2">
                  <Circle size={12} className="text-slate-300 shrink-0" />
                  <span className="text-xs text-slate-400 italic">Waiting to start...</span>
                </div>
              </div>
            )}

            {/* Task cards */}
            {cat.items.map((task) => {
              const prog = taskProgress(task, agentTaskProgress);
              const files = taskModifiedFiles[task.task_id];
              return (
                <div
                  key={task.task_id}
                  className={`border p-2.5 shadow-sm transition-all cursor-pointer hover:border-slate-300 ${
                    task.status === "completed"
                      ? `${cat.bg} ${cat.border}`
                      : "bg-white border-slate-200"
                  }`}
                  onClick={() => onSelectTask(task)}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <span className="text-[9px] font-mono text-slate-400 block">{task.task_id}</span>
                      <span className="text-xs font-semibold text-slate-700 leading-tight block">{task.title}</span>
                    </div>
                    {statusIcon(task.status)}
                  </div>
                  {task.assignee_agent_id && (
                    <div className="mt-1.5 text-[10px] font-mono text-slate-500 bg-slate-50 px-1.5 py-0.5 border border-slate-200 inline-block">
                      {resolveAgentName(task.assignee_agent_id)}
                    </div>
                  )}
                  {/* Modified files */}
                  {files && files.length > 0 && (
                    <div className="mt-1.5 space-y-0.5">
                      {files.map((fp) => (
                        <button
                          key={fp}
                          className="flex items-center gap-1 text-[10px] text-blue-600 hover:text-blue-800 hover:underline truncate max-w-full text-left"
                          title={fp}
                          onClick={(e) => { e.stopPropagation(); onSelectFile(fp); }}
                        >
                          <FolderOpen size={10} className="shrink-0 text-slate-400" />
                          <span className="truncate">{fp.split("/").pop()}</span>
                        </button>
                      ))}
                    </div>
                  )}
                  {/* Builder TODO progress */}
                  {prog && (
                    <div className="mt-2">
                      <div className="flex justify-between text-[9px] text-slate-400 mb-0.5">
                        <span>TODO Progress</span>
                        <span className="font-mono">{prog.done}/{prog.total} ({prog.pct}%)</span>
                      </div>
                      <div className="h-1 w-full bg-slate-100 overflow-hidden">
                        <div
                          className="h-full transition-all duration-500"
                          style={{
                            width: `${prog.pct}%`,
                            backgroundColor: prog.pct === 100 ? "#10b981" : cat.color,
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        );
      })}

      {/* Uncategorized tasks */}
      {uncategorized.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Wrench size={14} className="text-slate-500" />
            <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Other</span>
          </div>
          {uncategorized.map((task) => {
            const prog = taskProgress(task, agentTaskProgress);
            const files = taskModifiedFiles[task.task_id];
            return (
              <div
                key={task.task_id}
                className="bg-white border border-slate-200 p-2.5 shadow-sm cursor-pointer hover:border-slate-300"
                onClick={() => onSelectTask(task)}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <span className="text-[9px] font-mono text-slate-400 block">{task.task_id}</span>
                    <span className="text-xs font-semibold text-slate-700 leading-tight block">{task.title}</span>
                  </div>
                  {statusIcon(task.status)}
                </div>
                {task.assignee_agent_id && (
                  <div className="mt-1.5 text-[10px] font-mono text-slate-500 bg-slate-50 px-1.5 py-0.5 border border-slate-200 inline-block">
                    {resolveAgentName(task.assignee_agent_id)}
                  </div>
                )}
                {files && files.length > 0 && (
                  <div className="mt-1.5 space-y-0.5">
                    {files.map((fp) => (
                      <button
                        key={fp}
                        className="flex items-center gap-1 text-[10px] text-blue-600 hover:text-blue-800 hover:underline truncate max-w-full text-left"
                        title={fp}
                        onClick={(e) => { e.stopPropagation(); onSelectFile(fp); }}
                      >
                        <FolderOpen size={10} className="shrink-0 text-slate-400" />
                        <span className="truncate">{fp.split("/").pop()}</span>
                      </button>
                    ))}
                  </div>
                )}
                {prog && (
                  <div className="mt-2">
                    <div className="flex justify-between text-[9px] text-slate-400 mb-0.5">
                      <span>TODO Progress</span>
                      <span className="font-mono">{prog.done}/{prog.total} ({prog.pct}%)</span>
                    </div>
                    <div className="h-1 w-full bg-slate-100 overflow-hidden">
                      <div
                        className="h-full bg-slate-400 transition-all duration-500"
                        style={{ width: `${prog.pct}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
