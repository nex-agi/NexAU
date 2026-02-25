import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { parse as parsePartialJson } from "partial-json";
import type { AgentState } from "../types";
import { AskUserBlock, parseAskUserQuestions } from "./AskUserBlock";
import { WriteTodosBlock } from "./WriteTodosBlock";
import { MarkdownRenderer } from "./MarkdownRenderer";

interface AgentPanelProps {
  agent: AgentState;
  onSendMessage?: (content: string, toAgentId: string) => void;
}

export function AgentPanel({ agent, onSendMessage }: AgentPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const isAtBottomRef = useRef(true);
  const [submittedAskIds, setSubmittedAskIds] = useState<Set<string>>(new Set());

  // 从历史 blocks 中推导已回答的 ask_user block IDs：
  // ask_user tool_call（有 result）后紧跟 user_message 说明用户已作答
  const historyAnsweredAskIds = useMemo(() => {
    const ids = new Set<string>();
    const blocks = agent.blocks;
    for (let i = 0; i < blocks.length; i++) {
      const block = blocks[i];
      if (
        block.kind === "tool_call" &&
        block.name === "ask_user" &&
        block.result !== null
      ) {
        const blockId = block.id || `tc-${i}`;
        for (let j = i + 1; j < blocks.length; j++) {
          const next = blocks[j];
          if (next.kind === "user_message") {
            ids.add(blockId);
            break;
          }
          if (next.kind === "tool_call" && next.name === "ask_user") {
            break;
          }
        }
      }
    }
    return ids;
  }, [agent.blocks]);

  const markAskSubmitted = useCallback((blockId: string) => {
    setSubmittedAskIds((prev) => new Set(prev).add(blockId));
  }, []);

  const handleScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    // 距离底部 30px 以内视为"在底部"
    isAtBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < 30;
  }, []);

  useEffect(() => {
    if (isAtBottomRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [agent.blocks]);

  return (
    <div ref={containerRef} onScroll={handleScroll} style={styles.root}>
      {agent.blocks.map((block, i) => {
        switch (block.kind) {
          case "text":
            return (
              <div key={`t-${i}`}>
                <MarkdownRenderer content={block.content} />
              </div>
            );
          case "thinking":
            return (
              <ThinkingBlock key={`th-${i}`} content={block.content} />
            );
          case "tool_call": {
            // Render ask_user tool calls as interactive question UI
            // 使用 partial-json 支持流式渲染 ask_user 参数
            if (block.name === "ask_user" && onSendMessage) {
              const questions = parseAskUserQuestions(block.args, !block.argsDone);
              const blockId = block.id || `tc-${i}`;
              if (questions && questions.length > 0) {
                return (
                  <AskUserBlock
                    key={blockId}
                    questions={questions}
                    agentId={agent.agentId}
                    answered={submittedAskIds.has(blockId) || historyAnsweredAskIds.has(blockId)}
                    onSendMessage={onSendMessage}
                    onSubmitted={() => markAskSubmitted(blockId)}
                  />
                );
              }
            }
            // Render write_todos tool calls as todo list UI
            if (block.name === "write_todos") {
              return (
                <WriteTodosBlock
                  key={block.id || `tc-${i}`}
                  args={block.args}
                  argsDone={block.argsDone}
                  result={block.result}
                />
              );
            }
            return (
              <ToolCallBlock
                key={block.id || `tc-${i}`}
                name={block.name}
                args={block.args}
                argsDone={block.argsDone}
                result={block.result}
              />
            );
          }
          case "user_message":
            return (
              <div key={`um-${i}`} style={styles.userMsg}>
                <span style={styles.msgLabel}>User</span>
                {block.content}
              </div>
            );
          case "team_message":
            return (
              <div key={`tm-${i}`} style={styles.teamMsg}>
                <span style={styles.msgLabel}>{block.from}</span>
                {block.content}
              </div>
            );
          case "error":
            return (
              <div key={`err-${i}`} style={styles.errorMsg}>
                <span style={styles.errorLabel}>Error</span>
                {block.message}
              </div>
            );
        }
      })}
      <div ref={bottomRef} />
    </div>
  );
}

function tryParseJson(text: string, partial = false): unknown | null {
  if (!text) return null;
  try {
    return partial ? parsePartialJson(text) : JSON.parse(text);
  } catch {
    return null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function toOneLine(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function truncateOneLine(text: string, max: number): string {
  const one = toOneLine(text);
  if (one.length <= max) return one;
  return one.slice(0, Math.max(0, max - 1)) + "…";
}

function truncateMiddle(text: string, max: number): string {
  if (text.length <= max) return text;
  const head = Math.max(0, Math.floor(max * 0.6));
  const tail = Math.max(0, max - head - 1);
  return text.slice(0, head) + "…" + text.slice(text.length - tail);
}

function toolTitle(toolNameRaw: string): string {
  const toolName = toolNameRaw.trim();
  const titles: Record<string, string> = {
    read_file: "Read file",
    list_directory: "List directory",
    search_file_content: "Search",
    glob: "Find files",
    run_shell_command: "Run command",
  };
  return (titles[toolName] ?? toolName) || "Tool";
}

function buildInputPreview(
  toolNameRaw: string,
  argsObj: Record<string, unknown> | null,
  rawArgs: string
): string {
  const toolName = toolNameRaw.trim();
  const fallback = rawArgs ? truncateOneLine(rawArgs, 80) : "";
  if (!argsObj) return fallback;

  if (toolName === "read_file") {
    const filePath = asString(argsObj.file_path) ?? "";
    const offset = asNumber(argsObj.offset);
    const limit = asNumber(argsObj.limit);
    const pathPreview = filePath ? truncateMiddle(filePath, 60) : fallback;
    const range =
      offset !== null || limit !== null
        ? ` (offset ${offset ?? "—"}, limit ${limit ?? "—"})`
        : "";
    return `${pathPreview}${range}`.trim();
  }

  if (toolName === "list_directory") {
    const dirPath = asString(argsObj.dir_path) ?? "";
    const ignore = Array.isArray(argsObj.ignore) ? argsObj.ignore : null;
    const ignoreCount = ignore ? ignore.length : 0;
    const base = dirPath ? truncateMiddle(dirPath, 60) : fallback;
    return ignoreCount > 0 ? `${base} (ignore ${ignoreCount})` : base;
  }

  if (toolName === "search_file_content") {
    const pattern = asString(argsObj.pattern) ?? "";
    const dirPath = asString(argsObj.dir_path) ?? ".";
    const include = asString(argsObj.include);
    const p = pattern ? truncateOneLine(pattern, 40) : "(pattern)";
    const loc = dirPath ? truncateMiddle(dirPath, 40) : ".";
    const inc = include ? ` (include ${truncateOneLine(include, 30)})` : "";
    return `${p} in ${loc}${inc}`;
  }

  if (toolName === "glob") {
    const pattern = asString(argsObj.pattern) ?? "";
    const dirPath = asString(argsObj.dir_path) ?? ".";
    const p = pattern ? truncateOneLine(pattern, 48) : "(pattern)";
    const loc = dirPath ? truncateMiddle(dirPath, 40) : ".";
    return `${p} in ${loc}`;
  }

  if (toolName === "run_shell_command") {
    const command = asString(argsObj.command) ?? "";
    const dirPath = asString(argsObj.dir_path);
    const cmd = command ? truncateOneLine(command, 72) : "(command)";
    const loc = dirPath ? ` [in ${truncateMiddle(dirPath, 28)}]` : "";
    return `${cmd}${loc}`;
  }

  return fallback;
}

function buildInputRows(toolNameRaw: string, argsObj: Record<string, unknown> | null) {
  if (!argsObj) return [];
  const toolName = toolNameRaw.trim();

  const rows: Array<{ key: string; value: string }> = [];
  const add = (key: string, v: unknown) => {
    if (v === undefined) return;
    if (v === null) {
      rows.push({ key, value: "null" });
      return;
    }
    if (typeof v === "string") {
      rows.push({ key, value: v });
      return;
    }
    if (typeof v === "number" || typeof v === "boolean") {
      rows.push({ key, value: String(v) });
      return;
    }
    rows.push({ key, value: JSON.stringify(v) });
  };

  if (toolName === "read_file") {
    add("file_path", argsObj.file_path);
    add("offset", argsObj.offset);
    add("limit", argsObj.limit);
    return rows;
  }

  if (toolName === "list_directory") {
    add("dir_path", argsObj.dir_path);
    add("ignore", argsObj.ignore);
    add("file_filtering_options", argsObj.file_filtering_options);
    return rows;
  }

  if (toolName === "search_file_content") {
    add("pattern", argsObj.pattern);
    add("dir_path", argsObj.dir_path);
    add("include", argsObj.include);
    return rows;
  }

  if (toolName === "glob") {
    add("pattern", argsObj.pattern);
    add("dir_path", argsObj.dir_path);
    add("case_sensitive", argsObj.case_sensitive);
    add("respect_git_ignore", argsObj.respect_git_ignore);
    add("respect_gemini_ignore", argsObj.respect_gemini_ignore);
    return rows;
  }

  if (toolName === "run_shell_command") {
    add("command", argsObj.command);
    add("dir_path", argsObj.dir_path);
    add("description", argsObj.description);
    return rows;
  }

  for (const [k, v] of Object.entries(argsObj)) add(k, v);
  return rows;
}

function ThinkingBlock({ content }: { content: string }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div style={styles.thinkingBlock}>
      <div
        style={styles.thinkingHeader}
        onClick={() => setExpanded((v) => !v)}
      >
        <span style={styles.thinkingBadge}>THINKING</span>
        <span style={styles.thinkingPreview}>
          {expanded ? "" : truncateOneLine(content, 80) || "…"}
        </span>
        <span style={styles.chevron}>{expanded ? "▾" : "▸"}</span>
      </div>
      {expanded && (
        <div style={styles.thinkingBody}>
          <MarkdownRenderer content={content} />
        </div>
      )}
    </div>
  );
}

function ToolCallBlock({
  name,
  args,
  argsDone,
  result,
}: {
  name: string;
  args: string;
  argsDone: boolean;
  result: string | null;
}) {
  const [expanded, setExpanded] = useState(false);
  const [outputExpanded, setOutputExpanded] = useState(false);

  // 使用 partial-json 解析流式参数，支持部分结果渲染
  const parsedArgs = tryParseJson(args, !argsDone);
  const argsObj = isRecord(parsedArgs) ? parsedArgs : null;
  const prettyArgs = argsObj ? JSON.stringify(argsObj, null, 2) : args;

  // DEBUG: 打印 ToolCallBlock 数据，帮助定位输入/输出渲染错乱问题
  if (argsDone) {
    console.log(
      `[ToolCallBlock DEBUG] name=${name}`,
      `argsLen=${args.length}`,
      `argsObjKeys=${argsObj ? Object.keys(argsObj).join(",") : "NULL"}`,
      `resultLen=${result?.length ?? "null"}`,
      `prettyArgsLen=${prettyArgs.length}`,
    );
  }

  const parsedResult = result ? tryParseJson(result) : null;
  const outputObj = isRecord(parsedResult) ? parsedResult : null;

  const returnDisplay = outputObj ? asString(outputObj.returnDisplay) : null;
  const errorMessage = (() => {
    if (!outputObj) return null;
    const err = outputObj.error;
    if (!isRecord(err)) return null;
    return asString(err.message);
  })();
  const durationMs = outputObj ? asNumber(outputObj.duration_ms) : null;

  const outputPreview = (() => {
    if (!result) return argsDone ? "Running…" : "Waiting…";
    if (errorMessage) return `Error: ${truncateOneLine(errorMessage, 80)}`;
    if (returnDisplay) return truncateOneLine(returnDisplay, 100);
    const content = outputObj?.content;
    if (typeof content === "string") return truncateOneLine(content, 100);
    return "Done";
  })();

  const inputPreview = buildInputPreview(name, argsObj, args);
  const inputRows = buildInputRows(name, argsObj);

  const outputContentText = (() => {
    if (!result) return "";
    if (outputObj && "content" in outputObj) {
      const c = outputObj.content;
      if (typeof c === "string") return c;
      return JSON.stringify(c, null, 2);
    }
    if (typeof parsedResult === "string") return parsedResult;
    if (parsedResult === null) return result;
    return JSON.stringify(parsedResult, null, 2);
  })();

  return (
    <div style={styles.toolCall}>
      <div style={styles.toolHeader} onClick={() => setExpanded((v) => !v)}>
        <span style={styles.toolBadge}>TOOL</span>
        <span style={styles.toolTitle}>{toolTitle(name)}</span>
        {inputPreview && <span style={styles.toolPreview}>{inputPreview}</span>}
        <span style={styles.toolArrow}>→</span>
        <span
          style={{
            ...styles.toolPreview,
            color: errorMessage ? "#FF6B6B" : "#A09890",
          }}
        >
          {outputPreview}
        </span>
        <span style={styles.chevron}>{expanded ? "▾" : "▸"}</span>
      </div>

      {expanded && (
        <div style={styles.toolBody}>
          <div style={styles.toolSection}>
            <div style={styles.sectionLabel}>Input</div>
            {inputRows.length > 0 && (
              <div style={styles.kvGrid}>
                {inputRows.map((r) => (
                  <div key={r.key} style={styles.kvRow}>
                    <div style={styles.kvKey}>{r.key}</div>
                    <div style={styles.kvValue}>{r.value}</div>
                  </div>
                ))}
              </div>
            )}
            <pre style={styles.codeBlock}>{prettyArgs || "(empty)"}</pre>
          </div>

          <div style={styles.toolDivider} />

          <div style={styles.toolSection}>
            <div
              style={styles.outputHeader}
              onClick={() => setOutputExpanded((v) => !v)}
            >
              <div style={styles.sectionLabel}>Output</div>
              <div
                style={{
                  ...styles.outputPreview,
                  color: errorMessage ? "#FF6B6B" : "#A09890",
                }}
              >
                {outputPreview}
              </div>
              <span style={styles.chevron}>{outputExpanded ? "▾" : "▸"}</span>
            </div>

            {outputExpanded && !result && (
              <div style={styles.metaLine}>
                {argsDone
                  ? "Tool is running — output will appear here."
                  : "Waiting for args…"}
              </div>
            )}

            {outputExpanded && result && (
              <>
                {returnDisplay && (
                  <div style={styles.returnDisplay}>{returnDisplay}</div>
                )}
                {durationMs !== null && (
                  <div style={styles.metaLine}>duration: {durationMs}ms</div>
                )}
                {errorMessage && (
                  <div style={styles.metaLine}>error: {errorMessage}</div>
                )}
                <pre style={styles.codeBlock}>
                  {outputContentText || "(empty)"}
                </pre>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    fontSize: 13,
    lineHeight: 1.5,
    color: "#2D2A26",
    flex: 1,
    overflow: "auto",
  },
  text: {
    whiteSpace: "pre-wrap" as const,
    wordBreak: "break-word" as const,
  },
  thinkingBlock: {
    margin: "8px 0",
    borderRadius: 6,
    background: "#F5F0E8",
    border: "1px solid #E8E0D4",
    overflow: "hidden",
  },
  thinkingHeader: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "6px 10px",
    cursor: "pointer",
    userSelect: "none" as const,
  },
  thinkingBadge: {
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 0.6,
    padding: "2px 6px",
    borderRadius: 999,
    background: "#F0E6FF",
    border: "1px solid #D4C4F0",
    color: "#7C5CBF",
    flexShrink: 0,
  },
  thinkingPreview: {
    fontSize: 12,
    color: "#A09890",
    flex: 1,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap" as const,
  },
  thinkingBody: {
    borderTop: "1px solid #E8E0D4",
    padding: "10px 12px",
    fontSize: 12,
    color: "#6B6560",
  },
  toolCall: {
    margin: "8px 0",
    borderRadius: 6,
    background: "#FFFFFF",
    border: "1px solid #E8E0D4",
    overflow: "hidden",
  },
  toolHeader: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "6px 10px",
    cursor: "pointer",
    userSelect: "none" as const,
  },
  toolBadge: {
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 0.6,
    padding: "2px 6px",
    borderRadius: 999,
    background: "#E6F4FF",
    border: "1px solid #B8D8F0",
    color: "#4D96FF",
  },
  toolTitle: {
    fontSize: 12,
    fontWeight: 600,
    color: "#2D2A26",
    flexShrink: 0,
  },
  toolPreview: {
    fontSize: 12,
    color: "#A09890",
    flex: 1,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap" as const,
  },
  toolArrow: {
    color: "#C8C0B8",
    flexShrink: 0,
  },
  chevron: {
    fontSize: 10,
    color: "#A09890",
    flexShrink: 0,
  },
  toolBody: {
    borderTop: "1px solid #E8E0D4",
    background: "#FAF8F5",
    padding: "10px 12px",
    display: "flex",
    flexDirection: "column" as const,
    gap: 10,
  },
  toolSection: {
    display: "flex",
    flexDirection: "column" as const,
    gap: 8,
  },
  toolDivider: {
    height: 1,
    background: "#E8E0D4",
    margin: "4px 0",
  },
  sectionLabel: {
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: 0.6,
    textTransform: "uppercase" as const,
    color: "#A09890",
    padding: "2px 6px",
    borderRadius: 3,
    background: "#F5F0E8",
  },
  kvGrid: {
    display: "grid",
    gridTemplateColumns: "max-content 1fr",
    columnGap: 10,
    rowGap: 6,
    padding: "8px 10px",
    borderRadius: 6,
    border: "1px solid #E8E0D4",
    background: "#FFFFFF",
  },
  kvRow: {
    display: "contents",
  },
  kvKey: {
    fontSize: 11,
    color: "#A09890",
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace',
  },
  kvValue: {
    fontSize: 11,
    color: "#2D2A26",
    whiteSpace: "pre-wrap" as const,
    wordBreak: "break-word" as const,
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace',
  },
  outputHeader: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    cursor: "pointer",
    userSelect: "none" as const,
  },
  outputPreview: {
    flex: 1,
    fontSize: 12,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap" as const,
  },
  returnDisplay: {
    padding: "8px 10px",
    borderRadius: 6,
    background: "#FFFFFF",
    border: "1px solid #E8E0D4",
    color: "#2D2A26",
    fontSize: 12,
    whiteSpace: "pre-wrap" as const,
    wordBreak: "break-word" as const,
  },
  metaLine: {
    fontSize: 11,
    color: "#A09890",
  },
  codeBlock: {
    margin: 0,
    padding: "10px 12px",
    fontSize: 11,
    lineHeight: 1.4,
    color: "#2D2A26",
    background: "#F5F0E8",
    border: "1px solid #E8E0D4",
    borderRadius: 6,
    overflowX: "auto" as const,
    maxHeight: 360,
  },
  userMsg: {
    margin: "8px 0",
    padding: "6px 10px",
    borderRadius: 6,
    background: "#E8F5E9",
    border: "1px solid #C8E6C9",
    fontSize: 13,
    color: "#2E7D32",
    whiteSpace: "pre-wrap" as const,
    wordBreak: "break-word" as const,
  },
  teamMsg: {
    margin: "8px 0",
    padding: "6px 10px",
    borderRadius: 6,
    background: "#F0E6FF",
    border: "1px solid #D4C4F0",
    fontSize: 13,
    color: "#5E35B1",
    whiteSpace: "pre-wrap" as const,
    wordBreak: "break-word" as const,
  },
  msgLabel: {
    display: "inline-block",
    marginRight: 8,
    fontSize: 11,
    fontWeight: 600,
    color: "#A09890",
    textTransform: "uppercase" as const,
  },
  errorMsg: {
    margin: "8px 0",
    padding: "6px 10px",
    borderRadius: 6,
    background: "#FFF0F0",
    border: "1px solid #FFD0D0",
    fontSize: 13,
    color: "#FF6B6B",
    whiteSpace: "pre-wrap" as const,
    wordBreak: "break-word" as const,
  },
  errorLabel: {
    display: "inline-block",
    marginRight: 8,
    fontSize: 11,
    fontWeight: 600,
    color: "#FF6B6B",
    textTransform: "uppercase" as const,
  },
};
