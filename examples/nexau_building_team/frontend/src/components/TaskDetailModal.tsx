import {
  X, Circle, Loader2, CheckCircle2, Clock, User, FileText,
} from "lucide-react";
import type { TaskInfo } from "../types";
import { MarkdownRenderer } from "./MarkdownRenderer";

interface TaskDetailModalProps {
  task: TaskInfo;
  resolveAgentName: (agentId: string) => string;
  onClose: () => void;
}

function formatTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

function StatusBadge({ status }: { status: string }) {
  if (status === "completed") {
    return (
      <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-600 bg-emerald-50 border border-emerald-200 px-2 py-0.5">
        <CheckCircle2 size={12} /> Completed
      </span>
    );
  }
  if (status === "in_progress") {
    return (
      <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-blue-600 bg-blue-50 border border-blue-200 px-2 py-0.5">
        <Loader2 size={12} className="animate-spin" /> In Progress
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-slate-500 bg-slate-50 border border-slate-200 px-2 py-0.5">
      <Circle size={12} /> Pending
    </span>
  );
}

export function TaskDetailModal({ task, resolveAgentName, onClose }: TaskDetailModalProps) {
  return (
    <div className="fixed inset-0 bg-black/20 z-50 flex items-center justify-center" onClick={onClose}>
      <div
        className="bg-white border border-slate-200 shadow-2xl w-[520px] max-h-[80vh] flex flex-col animate-in zoom-in-95"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-5 pb-3 border-b border-slate-100 shrink-0">
          <div className="flex items-center gap-2">
            <FileText size={14} className="text-slate-500 shrink-0" />
            <span className="text-[10px] font-mono text-slate-400">{task.task_id}</span>
            <StatusBadge status={task.status} />
          </div>
          <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors">
            <X size={14} />
          </button>
        </div>

        {/* Body — scrollable */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {/* Title */}
          <h3 className="text-sm font-bold text-slate-800 leading-snug">{task.title}</h3>

          {/* Description */}
          {task.description && (
            <p className="text-xs text-slate-600 leading-relaxed">{task.description}</p>
          )}

          {/* Meta grid */}
          <div className="grid grid-cols-2 gap-3">
            {/* Assignee */}
            <div className="bg-slate-50 border border-slate-200 p-2.5">
              <div className="flex items-center gap-1.5 text-[10px] text-slate-400 uppercase tracking-wider font-semibold mb-1">
                <User size={10} /> Assignee
              </div>
              <p className="text-xs font-mono text-slate-700">
                {task.assignee_agent_id ? resolveAgentName(task.assignee_agent_id) : "—"}
              </p>
            </div>

            {/* Priority */}
            <div className="bg-slate-50 border border-slate-200 p-2.5">
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold mb-1">Priority</div>
              <p className="text-xs font-mono text-slate-700">
                {task.priority === 2 ? "Critical" : task.priority === 1 ? "High" : "Normal"}
              </p>
            </div>

            {/* Created */}
            <div className="bg-slate-50 border border-slate-200 p-2.5">
              <div className="flex items-center gap-1.5 text-[10px] text-slate-400 uppercase tracking-wider font-semibold mb-1">
                <Clock size={10} /> Created
              </div>
              <p className="text-xs font-mono text-slate-700">{formatTime(task.created_at)}</p>
            </div>

            {/* Updated */}
            <div className="bg-slate-50 border border-slate-200 p-2.5">
              <div className="flex items-center gap-1.5 text-[10px] text-slate-400 uppercase tracking-wider font-semibold mb-1">
                <Clock size={10} /> Updated
              </div>
              <p className="text-xs font-mono text-slate-700">{formatTime(task.updated_at)}</p>
            </div>
          </div>

          {/* Dependencies */}
          {task.dependencies.length > 0 && (
            <div>
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold mb-1.5">Dependencies</div>
              <div className="flex flex-wrap gap-1.5">
                {task.dependencies.map((dep) => (
                  <span key={dep} className="text-[10px] font-mono bg-slate-100 border border-slate-200 px-2 py-0.5 text-slate-600">{dep}</span>
                ))}
              </div>
            </div>
          )}

          {/* Result / Report */}
          {task.result_summary && (
            <div>
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold mb-1.5">Deliverable Report</div>
              <div className="bg-slate-50 border border-slate-200 p-3 text-xs text-slate-700 max-h-64 overflow-y-auto">
                <MarkdownRenderer content={task.result_summary} />
              </div>
            </div>
          )}

          {/* No result yet */}
          {!task.result_summary && task.status !== "completed" && (
            <p className="text-xs text-slate-400 italic">No deliverable yet — task is still {task.status === "in_progress" ? "in progress" : "pending"}.</p>
          )}
        </div>
      </div>
    </div>
  );
}
