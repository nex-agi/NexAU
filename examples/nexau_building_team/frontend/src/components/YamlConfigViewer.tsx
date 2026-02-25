import yaml from "js-yaml";

interface YamlConfigViewerProps {
  content: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Rec = Record<string, any>;

function parseYaml(content: string): Rec | null {
  try {
    const doc = yaml.load(content);
    return doc && typeof doc === "object" ? (doc as Rec) : null;
  } catch {
    return null;
  }
}

function detectConfigType(doc: Rec): "agent" | "tool" | null {
  if (doc.type === "agent") return "agent";
  if (doc.type === "tool") return "tool";
  // heuristic: has llm_config → agent
  if (doc.llm_config || doc.system_prompt || doc.sub_agents || doc.skills) return "agent";
  // heuristic: has input_schema → tool
  if (doc.input_schema) return "tool";
  return null;
}

export function detectYamlConfig(content: string) {
  const doc = parseYaml(content);
  if (!doc) return null;
  const configType = detectConfigType(doc);
  if (!configType) return null;
  return { doc, configType };
}

export function YamlConfigViewer({ content }: YamlConfigViewerProps) {
  const parsed = detectYamlConfig(content);
  if (!parsed) return <div style={s.empty}>Unable to parse config</div>;
  const { doc, configType } = parsed;

  return configType === "agent" ? (
    <AgentConfigView doc={doc} />
  ) : (
    <ToolConfigView doc={doc} />
  );
}

/* ── Shared helpers ─────────────────────────────────────── */

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={s.section}>
      <div style={s.sectionTitle}>{title}</div>
      <div style={s.sectionBody}>{children}</div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  if (value === undefined || value === null || value === "") return null;
  return (
    <div style={s.field}>
      <span style={s.fieldLabel}>{label}</span>
      <span style={s.fieldValue}>{typeof value === "boolean" ? (value ? "Yes" : "No") : String(value)}</span>
    </div>
  );
}

function Badge({ text, color }: { text: string; color: string }) {
  return (
    <span style={{ ...s.badge, background: color }}>{text}</span>
  );
}

/* ── Agent Config View ──────────────────────────────────── */

function AgentConfigView({ doc }: { doc: Rec }) {
  const llm = doc.llm_config || {};
  const sandbox = doc.sandbox_config;
  const tools: Rec[] = doc.tools || [];
  const skills: (string | Rec)[] = doc.skills || [];
  const subAgents: Rec[] = doc.sub_agents || [];
  const mcpServers: Rec[] = doc.mcp_servers || [];
  const middlewares: Rec[] = doc.middlewares || [];
  const stopTools: string[] = doc.stop_tools || [];

  return (
    <div style={s.root}>
      {/* Header */}
      <div style={s.header}>
        <div style={s.headerTop}>
          <Badge text="AGENT" color="#A78BFA" />
          <span style={s.headerName}>{doc.name || "Unnamed Agent"}</span>
        </div>
        {doc.description && <div style={s.headerDesc}>{doc.description}</div>}
      </div>

      {/* LLM Config */}
      <Section title="LLM Configuration">
        <Field label="Model" value={llm.model} />
        <Field label="API Type" value={llm.api_type} />
        <Field label="Base URL" value={llm.base_url} />
        <Field label="Temperature" value={llm.temperature} />
        <Field label="Max Tokens" value={llm.max_tokens} />
        <Field label="Stream" value={llm.stream} />
      </Section>

      {/* System Prompt */}
      {doc.system_prompt && (
        <Section title="System Prompt">
          <Field label="Type" value={doc.system_prompt_type || "string"} />
          <div style={s.promptPreview}>
            {String(doc.system_prompt).length > 300
              ? String(doc.system_prompt).slice(0, 300) + "…"
              : String(doc.system_prompt)}
          </div>
        </Section>
      )}

      {/* Tools */}
      {tools.length > 0 && (
        <Section title={`Tools (${tools.length})`}>
          <div style={s.table}>
            <div style={{ ...s.tableRow, ...s.tableHeader }}>
              <span style={{ ...s.tableCell, flex: 1 }}>Name</span>
              <span style={{ ...s.tableCell, flex: 2 }}>Source</span>
              <span style={{ ...s.tableCell, flex: 0.5 }}>Flags</span>
            </div>
            {tools.map((t, i) => (
              <div key={i} style={s.tableRow}>
                <span style={{ ...s.tableCell, flex: 1, color: "#7C5CBF" }}>{t.name || "—"}</span>
                <span style={{ ...s.tableCell, flex: 2, color: "#A09890", fontSize: 11 }}>
                  {t.yaml_path || t.binding || "—"}
                </span>
                <span style={{ ...s.tableCell, flex: 0.5 }}>
                  {t.lazy && <Badge text="lazy" color="#C8C0B8" />}
                  {t.as_skill && <Badge text="skill" color="#6BCB77" />}
                </span>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Skills */}
      {skills.length > 0 && (
        <Section title={`Skills (${skills.length})`}>
          {skills.map((sk, i) => {
            const path = typeof sk === "string" ? sk : sk.path || sk.name || "—";
            const name = path.split("/").filter(Boolean).pop() || path;
            return (
              <div key={i} style={s.listItem}>
                <Badge text="skill" color="#6BCB77" />
                <span style={{ color: "#7C5CBF", marginLeft: 6 }}>{name}</span>
                <span style={{ color: "#A09890", fontSize: 11, marginLeft: 8, fontFamily: "monospace" }}>
                  {path}
                </span>
              </div>
            );
          })}
        </Section>
      )}

      {/* Sub-agents */}
      {subAgents.length > 0 && (
        <Section title={`Sub-agents (${subAgents.length})`}>
          {subAgents.map((a, i) => (
            <div key={i} style={s.listItem}>
              <span style={{ color: "#7C5CBF" }}>{a.name}</span>
              <span style={{ color: "#A09890", fontSize: 11, marginLeft: 8 }}>{a.config_path}</span>
            </div>
          ))}
        </Section>
      )}

      {/* Sandbox */}
      {sandbox && (
        <Section title="Sandbox">
          <Field label="Type" value={sandbox.type} />
          <Field label="Work Dir" value={sandbox._work_dir || sandbox.work_dir} />
        </Section>
      )}

      {/* MCP Servers */}
      {mcpServers.length > 0 && (
        <Section title={`MCP Servers (${mcpServers.length})`}>
          {mcpServers.map((m, i) => (
            <div key={i} style={s.listItem}>
              <Badge text={m.type || "stdio"} color="#C8C0B8" />
              <span style={{ color: "#7C5CBF", marginLeft: 6 }}>{m.name}</span>
              <span style={{ color: "#A09890", fontSize: 11, marginLeft: 8 }}>
                {m.command || m.url || ""}
              </span>
            </div>
          ))}
        </Section>
      )}

      {/* Execution Limits */}
      <Section title="Execution">
        <Field label="Max Iterations" value={doc.max_iterations} />
        <Field label="Timeout" value={doc.timeout ? `${doc.timeout}s` : undefined} />
        <Field label="Max Context Tokens" value={doc.max_context_tokens} />
        <Field label="Retry Attempts" value={doc.retry_attempts} />
        <Field label="Tool Call Mode" value={doc.tool_call_mode} />
        {stopTools.length > 0 && (
          <Field label="Stop Tools" value={stopTools.join(", ")} />
        )}
      </Section>

      {/* Middlewares */}
      {middlewares.length > 0 && (
        <Section title={`Middlewares (${middlewares.length})`}>
          {middlewares.map((m, i) => (
            <div key={i} style={s.listItem}>
              <span style={{ color: "#A09890", fontSize: 12, fontFamily: "monospace" }}>{m.import}</span>
            </div>
          ))}
        </Section>
      )}
    </div>
  );
}

/* ── Tool Config View ───────────────────────────────────── */

function ToolConfigView({ doc }: { doc: Rec }) {
  const schema = doc.input_schema || {};
  const props: Rec = schema.properties || {};
  const required: string[] = schema.required || [];
  const paramNames = Object.keys(props);

  return (
    <div style={s.root}>
      {/* Header */}
      <div style={s.header}>
        <div style={s.headerTop}>
          <Badge text="TOOL" color="#6BCB77" />
          <span style={s.headerName}>{doc.name || "Unnamed Tool"}</span>
        </div>
        {doc.description && <div style={s.headerDesc}>{doc.description}</div>}
      </div>

      {/* Input Parameters */}
      {paramNames.length > 0 && (
        <Section title={`Input Parameters (${paramNames.length})`}>
          <div style={s.table}>
            <div style={{ ...s.tableRow, ...s.tableHeader }}>
              <span style={{ ...s.tableCell, flex: 1 }}>Name</span>
              <span style={{ ...s.tableCell, flex: 0.6 }}>Type</span>
              <span style={{ ...s.tableCell, flex: 2 }}>Description</span>
              <span style={{ ...s.tableCell, flex: 0.4 }}>Req</span>
            </div>
            {paramNames.map((name) => {
              const p = props[name] || {};
              return (
                <div key={name} style={s.tableRow}>
                  <span style={{ ...s.tableCell, flex: 1, color: "#7C5CBF", fontFamily: "monospace", fontSize: 12 }}>
                    {name}
                  </span>
                  <span style={{ ...s.tableCell, flex: 0.6 }}>
                    <Badge text={p.type || "any"} color="#C8C0B8" />
                  </span>
                  <span style={{ ...s.tableCell, flex: 2, color: "#6B6560", fontSize: 12 }}>
                    {p.description || "—"}
                  </span>
                  <span style={{ ...s.tableCell, flex: 0.4 }}>
                    {required.includes(name) ? "✓" : ""}
                  </span>
                </div>
              );
            })}
          </div>
        </Section>
      )}

      {/* Binding */}
      {(doc.binding || doc.builtin) && (
        <Section title="Binding">
          <Field label={doc.builtin ? "Builtin" : "Import"} value={doc.binding || doc.builtin} />
        </Section>
      )}

      {/* Options */}
      <Section title="Options">
        <Field label="Cache" value={doc.use_cache} />
        <Field label="Disable Parallel" value={doc.disable_parallel} />
        <Field label="Lazy" value={doc.lazy} />
      </Section>
    </div>
  );
}

/* ── Styles ─────────────────────────────────────────────── */

const s: Record<string, React.CSSProperties> = {
  empty: {
    padding: 24,
    color: "#A09890",
    textAlign: "center",
  },
  root: {
    padding: "16px 20px",
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  header: {
    padding: "16px 20px",
    background: "#FAF8F5",
    borderRadius: 8,
    border: "1px solid #E8E0D4",
  },
  headerTop: {
    display: "flex",
    alignItems: "center",
    gap: 10,
  },
  headerName: {
    fontSize: 18,
    fontWeight: 600,
    color: "#2D2A26",
  },
  headerDesc: {
    marginTop: 8,
    fontSize: 13,
    color: "#6B6560",
    lineHeight: 1.5,
  },
  badge: {
    display: "inline-block",
    padding: "2px 8px",
    borderRadius: 4,
    fontSize: 10,
    fontWeight: 700,
    color: "#fff",
    letterSpacing: "0.05em",
    textTransform: "uppercase" as const,
  },
  section: {
    background: "#FFFFFF",
    borderRadius: 8,
    border: "1px solid #E8E0D4",
    overflow: "hidden",
  },
  sectionTitle: {
    padding: "8px 16px",
    fontSize: 12,
    fontWeight: 600,
    color: "#A09890",
    textTransform: "uppercase" as const,
    letterSpacing: "0.06em",
    borderBottom: "1px solid #E8E0D4",
    background: "#FAF8F5",
  },
  sectionBody: {
    padding: "10px 16px",
  },
  field: {
    display: "flex",
    justifyContent: "space-between",
    padding: "5px 0",
    borderBottom: "1px solid #F0EBE3",
  },
  fieldLabel: {
    fontSize: 12,
    color: "#A09890",
  },
  fieldValue: {
    fontSize: 12,
    color: "#2D2A26",
    fontFamily: "monospace",
    textAlign: "right" as const,
    maxWidth: "60%",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  promptPreview: {
    marginTop: 8,
    padding: 12,
    background: "#F5F0E8",
    borderRadius: 6,
    border: "1px solid #E8E0D4",
    fontSize: 12,
    color: "#6B6560",
    lineHeight: 1.6,
    whiteSpace: "pre-wrap",
    fontFamily: "monospace",
    maxHeight: 200,
    overflow: "auto",
  },
  table: {
    display: "flex",
    flexDirection: "column",
  },
  tableRow: {
    display: "flex",
    alignItems: "center",
    padding: "6px 0",
    borderBottom: "1px solid #F0EBE3",
    gap: 8,
  },
  tableHeader: {
    fontWeight: 600,
    fontSize: 11,
    color: "#A09890",
    textTransform: "uppercase" as const,
    letterSpacing: "0.04em",
  },
  tableCell: {
    fontSize: 12,
    color: "#2D2A26",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  listItem: {
    display: "flex",
    alignItems: "center",
    padding: "5px 0",
    borderBottom: "1px solid #F0EBE3",
    fontSize: 12,
  },
};
