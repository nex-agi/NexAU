import { parse as parsePartialJson } from "partial-json";

interface TodoItem {
  description: string;
  status: "pending" | "in_progress" | "completed" | "cancelled";
}

interface WriteTodosBlockProps {
  args: string;
  argsDone: boolean;
  result: string | null;
}

/** Parse todos array from write_todos tool call args. */
export function parseTodos(
  argsJson: string,
  partial = false
): TodoItem[] | null {
  try {
    const parsed = partial ? parsePartialJson(argsJson) : JSON.parse(argsJson);
    if (parsed && typeof parsed === "object" && Array.isArray(parsed.todos)) {
      return parsed.todos as TodoItem[];
    }
  } catch {
    // args not yet complete
  }
  return null;
}

export function WriteTodosBlock({ args, argsDone, result }: WriteTodosBlockProps) {
  const todos = parseTodos(args, !argsDone);

  if (!todos || todos.length === 0) {
    return (
      <div style={styles.container}>
        <div style={styles.headerRow}>
          <span style={styles.badge}>TODO</span>
          <span style={styles.loadingLabel}>Loading tasks…</span>
        </div>
      </div>
    );
  }

  const completed = todos.filter((t) => t.status === "completed").length;
  const cancelled = todos.filter((t) => t.status === "cancelled").length;
  const active = todos.length - cancelled;
  const progressPct = active > 0 ? Math.round((completed / active) * 100) : 0;
  const allDone = completed === active && argsDone;

  return (
    <div style={styles.container}>
      <div style={styles.headerRow}>
        <span style={styles.badge}>TODO</span>
        <span style={styles.countLabel}>
          {completed}/{active}
        </span>
        <div style={styles.progressTrack}>
          <div
            style={{
              ...styles.progressFill,
              width: `${progressPct}%`,
              background: allDone ? "#6BCB77" : "#4D96FF",
            }}
          />
        </div>
      </div>

      <div style={styles.list}>
        {todos.map((todo, i) => (
          <div key={i} style={styles.item}>
            <span style={statusIconStyle(todo.status)}>
              {statusIcon(todo.status)}
            </span>
            <span
              style={{
                ...styles.itemText,
                ...(todo.status === "completed" ? styles.itemDone : {}),
                ...(todo.status === "in_progress" ? styles.itemActive : {}),
                ...(todo.status === "cancelled" ? styles.itemCancelled : {}),
              }}
            >
              {todo.description}
            </span>
            {todo.status === "in_progress" && (
              <span style={styles.statusDot} />
            )}
          </div>
        ))}
      </div>

      {result && (
        <div style={styles.footer}>
          {allDone ? "All tasks tracked" : "Tasks updated"}
        </div>
      )}
    </div>
  );
}

function statusIcon(status: TodoItem["status"]): string {
  switch (status) {
    case "completed":
      return "✓";
    case "in_progress":
      return "●";
    case "cancelled":
      return "✕";
    case "pending":
      return "○";
  }
}

function statusIconStyle(status: TodoItem["status"]): React.CSSProperties {
  const base: React.CSSProperties = {
    fontSize: 12,
    fontWeight: 700,
    width: 18,
    textAlign: "center",
    flexShrink: 0,
    lineHeight: "20px",
  };
  switch (status) {
    case "completed":
      return { ...base, color: "#6BCB77" };
    case "in_progress":
      return { ...base, color: "#4D96FF" };
    case "cancelled":
      return { ...base, color: "#C8C0B8" };
    case "pending":
      return { ...base, color: "#A09890" };
  }
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    margin: "8px 0",
    padding: "10px 14px",
    borderRadius: 8,
    background: "#FFFFFF",
    border: "1px solid #E8E0D4",
    display: "flex",
    flexDirection: "column",
    gap: 8,
  },
  headerRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  badge: {
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 0.6,
    padding: "2px 6px",
    borderRadius: 999,
    background: "#E6F4FF",
    border: "1px solid #B8D8F0",
    color: "#4D96FF",
  },
  countLabel: {
    fontSize: 12,
    color: "#A09890",
    fontWeight: 500,
  },
  loadingLabel: {
    fontSize: 12,
    color: "#A09890",
  },
  progressTrack: {
    flex: 1,
    height: 4,
    borderRadius: 2,
    background: "#F0EBE3",
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    borderRadius: 2,
    transition: "width 0.3s ease",
  },
  list: {
    display: "flex",
    flexDirection: "column",
    gap: 2,
  },
  item: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "4px 6px",
    borderRadius: 4,
  },
  itemText: {
    fontSize: 12,
    color: "#2D2A26",
    flex: 1,
  },
  itemDone: {
    color: "#C8C0B8",
    textDecoration: "line-through",
  },
  itemCancelled: {
    color: "#C8C0B8",
    textDecoration: "line-through",
    fontStyle: "italic",
  },
  itemActive: {
    color: "#2D2A26",
    fontWeight: 500,
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: "50%",
    background: "#4D96FF",
    flexShrink: 0,
    animation: "pulse 1.5s ease-in-out infinite",
  },
  footer: {
    fontSize: 11,
    color: "#A09890",
    textAlign: "right",
  },
};
