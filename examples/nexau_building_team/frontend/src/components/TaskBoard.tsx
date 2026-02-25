import { useCallback, useEffect, useState } from "react";
import type { TaskInfo } from "../types";

interface TaskBoardProps {
  userId: string;
  sessionId: string;
  isStreaming: boolean;
}

const COLLAPSED_MAX = 5;

const STATUS_PRIORITY: Record<string, number> = {
  in_progress: 0,
  pending: 1,
  completed: 2,
};

export function TaskBoard({ userId, sessionId, isStreaming }: TaskBoardProps) {
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [collapsed, setCollapsed] = useState(false);

  const fetchTasks = useCallback(async () => {
    try {
      const res = await fetch(
        `/team/tasks?user_id=${encodeURIComponent(userId)}&session_id=${encodeURIComponent(sessionId)}`
      );
      if (res.ok) {
        setTasks(await res.json());
      }
    } catch {
      // ignore fetch errors during polling
    }
  }, [userId, sessionId]);

  useEffect(() => {
    if (!isStreaming) return;
    fetchTasks();
    const interval = setInterval(fetchTasks, 3000);
    return () => clearInterval(interval);
  }, [isStreaming, fetchTasks]);

  if (tasks.length === 0) return null;

  const visible = collapsed
    ? [...tasks]
        .sort((a, b) => (STATUS_PRIORITY[a.status] ?? 3) - (STATUS_PRIORITY[b.status] ?? 3))
        .slice(0, COLLAPSED_MAX)
    : tasks;
  const hiddenCount = collapsed ? Math.max(0, tasks.length - COLLAPSED_MAX) : 0;

  return (
    <div style={styles.board}>
      <div
        style={styles.titleRow}
        onClick={() => setCollapsed((c) => !c)}
      >
        <span style={styles.title}>Task Board</span>
        <span style={styles.collapseHint}>
          {collapsed ? `▸ ${tasks.length} tasks` : "▾"}
        </span>
      </div>
      {!collapsed ? (
        <div style={styles.list}>
          {visible.map((t) => (
            <div key={t.task_id} style={styles.task}>
              <span style={{ ...styles.badge, ...statusStyle(t.status) }}>
                {t.status}
              </span>
              <span style={styles.taskTitle}>{t.title}</span>
              {t.assignee_agent_id && (
                <span style={styles.assignee}>→ {t.assignee_agent_id}</span>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div style={styles.list}>
          {visible.map((t) => (
            <div key={t.task_id} style={styles.task}>
              <span style={{ ...styles.badge, ...statusStyle(t.status) }}>
                {t.status}
              </span>
              <span style={styles.taskTitle}>{t.title}</span>
            </div>
          ))}
          {hiddenCount > 0 && (
            <div style={styles.moreHint}>+{hiddenCount} more…</div>
          )}
        </div>
      )}
    </div>
  );
}

function statusStyle(status: string): React.CSSProperties {
  switch (status) {
    case "completed":
      return { background: "#E8F5E9", color: "#2E7D32" };
    case "in_progress":
      return { background: "#FFF3E0", color: "#E65100" };
    default:
      return { background: "#F5F0E8", color: "#A09890" };
  }
}

const styles: Record<string, React.CSSProperties> = {
  board: {
    padding: "12px 16px",
    borderBottom: "1px solid #E8E0D4",
    background: "#FFFFFF",
  },
  titleRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    cursor: "pointer",
    userSelect: "none",
    marginBottom: 8,
  },
  title: {
    fontSize: 12,
    fontWeight: 600,
    color: "#A09890",
    textTransform: "uppercase" as const,
    letterSpacing: 1,
  },
  collapseHint: {
    fontSize: 11,
    color: "#C8C0B8",
  },
  list: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
  },
  task: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    fontSize: 13,
    color: "#2D2A26",
  },
  badge: {
    fontSize: 10,
    padding: "2px 6px",
    borderRadius: 4,
    fontWeight: 600,
    textTransform: "uppercase" as const,
    flexShrink: 0,
  },
  taskTitle: {
    flex: 1,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap" as const,
  },
  assignee: {
    fontSize: 11,
    color: "#A09890",
    flexShrink: 0,
  },
  moreHint: {
    fontSize: 11,
    color: "#C8C0B8",
    paddingTop: 4,
    textAlign: "center" as const,
  },
};
