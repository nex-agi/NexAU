import { useCallback, useEffect, useMemo, useState } from "react";
import { ChatInput } from "./components/ChatInput";
import { FileTree } from "./components/FileTree";
import { FileViewer } from "./components/FileViewer";
import { TaskBoard } from "./components/TaskBoard";
import { TeamLayout } from "./components/TeamLayout";
import { useTeamStream } from "./hooks/useTeamStream";

const USER_ID = "demo-user";
const SESSION_STORAGE_KEY = "nexau-session-id";

function getOrCreateSessionId(): string {
  const stored = localStorage.getItem(SESSION_STORAGE_KEY);
  if (stored) return stored;
  const id = `session-${Date.now()}`;
  localStorage.setItem(SESSION_STORAGE_KEY, id);
  return id;
}

export default function App() {
  const [sessionId, setSessionId] = useState(getOrCreateSessionId);
  const [sessions, setSessions] = useState<string[]>([]);
  const { agents, setAgents, isStreaming, error, startStream, sendUserMessage, stopStream, loadHistory, subscribeToStream } =
    useTeamStream();

  // 获取可用 session 列表（页面加载 + streaming 结束后刷新）
  const refreshSessions = useCallback(() => {
    fetch(`/team/sessions?user_id=${encodeURIComponent(USER_ID)}`)
      .then((r) => r.json())
      .then((list: string[]) => setSessions(list))
      .catch(() => {});
  }, []);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  // streaming 结束后刷新 session 列表（新 session 可能已创建）
  useEffect(() => {
    if (!isStreaming) refreshSessions();
  }, [isStreaming, refreshSessions]);

  // 页面加载或切换 session 时恢复历史事件，并在 team 仍在运行时重连 SSE 流
  useEffect(() => {
    loadHistory(USER_ID, sessionId).then((count) => {
      subscribeToStream(USER_ID, sessionId, count);
    });
  }, [loadHistory, subscribeToStream, sessionId]);

  const switchSession = useCallback(
    (newSessionId: string) => {
      if (newSessionId === sessionId || isStreaming) return;
      localStorage.setItem(SESSION_STORAGE_KEY, newSessionId);
      setAgents({});
      setSessionId(newSessionId);
    },
    [sessionId, isStreaming, setAgents],
  );

  const agentIds = useMemo(() => Object.keys(agents), [agents]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  const handleSend = useCallback(
    (message: string, toAgentId?: string) => {
      if (isStreaming && toAgentId) {
        sendUserMessage(message, toAgentId);
      } else {
        startStream(message, USER_ID, sessionId);
      }
    },
    [isStreaming, startStream, sendUserMessage, sessionId],
  );

  return (
    <div style={styles.root}>
      {/* Left panel: Agent Team (35%) */}
      <div style={styles.leftPanel}>
        <header style={styles.header}>
          <h1 style={styles.title}>Agent Team</h1>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            {sessions.length > 0 ? (
              <select
                value={sessionId}
                onChange={(e) => switchSession(e.target.value)}
                disabled={isStreaming}
                style={styles.sessionSelect}
              >
                {sessions.map((sid) => (
                  <option key={sid} value={sid}>
                    {sid}
                  </option>
                ))}
                {!sessions.includes(sessionId) && (
                  <option value={sessionId}>{sessionId}</option>
                )}
              </select>
            ) : (
              <span style={{ fontSize: 11, color: "#A09890" }}>{sessionId.slice(0, 20)}</span>
            )}
            <button
              onClick={() => {
                const newId = `session-${Date.now()}`;
                localStorage.setItem(SESSION_STORAGE_KEY, newId);
                setAgents({});
                setSessionId(newId);
              }}
              style={styles.newSessionBtn}
              disabled={isStreaming}
            >
              New Session
            </button>
            {isStreaming && (
              <button onClick={stopStream} style={styles.stopBtn}>
                Stop
              </button>
            )}
          </div>
        </header>

        {error && <div style={styles.error}>Error: {error}</div>}

        <TaskBoard
          userId={USER_ID}
          sessionId={sessionId}
          isStreaming={isStreaming}
        />

        <TeamLayout agents={agents} onSendMessage={handleSend} />

        <ChatInput
          onSend={handleSend}
          isStreaming={isStreaming}
          agentIds={agentIds}
        />
      </div>

      {/* Middle panel: File Viewer (50%) */}
      <div style={styles.middlePanel}>
        <FileViewer filePath={selectedFile} />
      </div>

      {/* Right panel: File Tree (15%) */}
      <div style={styles.rightPanel}>
        <FileTree
          selectedPath={selectedFile}
          onSelectFile={setSelectedFile}
          isStreaming={isStreaming}
        />
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    display: "flex",
    flexDirection: "row",
    height: "100vh",
    background: "#FBF8F3",
    color: "#2D2A26",
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  leftPanel: {
    width: "35%",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  middlePanel: {
    width: "50%",
    overflow: "hidden",
  },
  rightPanel: {
    width: "15%",
    overflow: "hidden",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "12px 16px",
    borderBottom: "1px solid #E8E0D4",
    background: "#FFFFFF",
    flexShrink: 0,
  },
  title: {
    margin: 0,
    fontSize: 18,
    fontWeight: 600,
    color: "#2D2A26",
  },
  stopBtn: {
    padding: "6px 14px",
    borderRadius: 6,
    border: "none",
    background: "#FF6B6B",
    color: "#FFFFFF",
    fontSize: 13,
    cursor: "pointer",
  },
  newSessionBtn: {
    padding: "6px 14px",
    borderRadius: 6,
    border: "1px solid #E8E0D4",
    background: "#FFFFFF",
    color: "#6B6560",
    fontSize: 12,
    cursor: "pointer",
  },
  sessionSelect: {
    padding: "4px 8px",
    borderRadius: 4,
    border: "1px solid #E8E0D4",
    background: "#FFFFFF",
    color: "#2D2A26",
    fontSize: 11,
    maxWidth: 180,
    cursor: "pointer",
  },
  error: {
    padding: "8px 16px",
    background: "#FFF0F0",
    color: "#FF6B6B",
    fontSize: 13,
  },
};
