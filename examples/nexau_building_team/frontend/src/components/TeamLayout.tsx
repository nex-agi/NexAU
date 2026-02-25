import { useState, useEffect } from "react";
import type { AgentState } from "../types";
import { AgentPanel } from "./AgentPanel";

interface TeamLayoutProps {
  agents: Record<string, AgentState>;
  onSendMessage?: (content: string, toAgentId: string) => void;
}

export function TeamLayout({ agents, onSendMessage }: TeamLayoutProps) {
  const entries = Object.values(agents);
  const [activeTab, setActiveTab] = useState<string | null>(null);

  // Auto-select first agent when agents appear
  useEffect(() => {
    if (activeTab === null && entries.length > 0) {
      setActiveTab(entries[0].agentId);
    }
  }, [activeTab, entries]);

  // Reset if active agent disappears
  useEffect(() => {
    if (activeTab && !agents[activeTab] && entries.length > 0) {
      setActiveTab(entries[0].agentId);
    }
  }, [activeTab, agents, entries]);

  if (entries.length === 0) {
    return (
      <div style={styles.empty}>
        Send a message to start the agent team.
      </div>
    );
  }

  const selected = activeTab ? agents[activeTab] : entries[0];

  return (
    <div style={styles.container}>
      <div style={styles.tabBar}>
        {entries.map((agent) => {
          const isActive = agent.agentId === (selected?.agentId ?? "");
          return (
            <button
              key={agent.agentId}
              onClick={() => setActiveTab(agent.agentId)}
              style={{
                ...styles.tab,
                ...(isActive ? styles.tabActive : {}),
              }}
            >
              <span
                style={{
                  ...styles.dot,
                  background: agent.isActive ? "#6BCB77" : "#C8C0B8",
                }}
              />
              <span style={styles.tabLabel}>{agent.agentId}</span>
              <span style={styles.tabRole}>{agent.roleName}</span>
            </button>
          );
        })}
      </div>

      <div style={styles.content}>
        {selected && <AgentPanel agent={selected} onSendMessage={onSendMessage} />}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    flex: 1,
    overflow: "hidden",
  },
  tabBar: {
    display: "flex",
    gap: 2,
    padding: "0 16px",
    background: "#FFFFFF",
    borderBottom: "1px solid #E8E0D4",
    overflowX: "auto",
    flexShrink: 0,
  },
  tab: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "8px 14px",
    border: "none",
    borderBottom: "2px solid transparent",
    background: "transparent",
    color: "#A09890",
    fontSize: 13,
    cursor: "pointer",
    whiteSpace: "nowrap" as const,
    transition: "color 0.15s, border-color 0.15s",
  },
  tabActive: {
    color: "#2D2A26",
    borderBottomColor: "#FF6B6B",
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: "50%",
    flexShrink: 0,
  },
  tabLabel: {
    fontWeight: 600,
  },
  tabRole: {
    fontSize: 10,
    textTransform: "uppercase" as const,
    letterSpacing: 0.5,
    opacity: 0.6,
  },
  content: {
    flex: 1,
    overflow: "hidden",
    padding: 16,
    display: "flex",
    flexDirection: "column" as const,
  },
  empty: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#A09890",
    fontSize: 15,
  },
};
