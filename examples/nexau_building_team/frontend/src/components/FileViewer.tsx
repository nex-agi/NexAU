import { useState, useEffect, useRef } from "react";
import hljs from "highlight.js";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import mermaid from "mermaid";
import yaml from "js-yaml";
import { detectYamlConfig, YamlConfigViewer } from "./YamlConfigViewer";

mermaid.initialize({ startOnLoad: false, theme: "default" });

let mermaidCounter = 0;

function MermaidBlock({ chart }: { chart: string }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const id = `mermaid-${++mermaidCounter}`;
    mermaid.render(id, chart).then(({ svg }) => {
      if (containerRef.current) containerRef.current.innerHTML = svg;
    }).catch(() => {
      // mermaid v11 会在 DOM 中残留错误元素，清理掉
      const errEl = document.getElementById(id);
      if (errEl) errEl.remove();
      // 语法错误时静默显示原始代码
      if (containerRef.current) {
        containerRef.current.textContent = chart;
        Object.assign(containerRef.current.style, {
          color: "#A09890",
          whiteSpace: "pre-wrap",
          fontSize: "12px",
          fontFamily: "monospace",
          padding: "10px 12px",
          background: "#F5F0E8",
          border: "1px solid #E8E0D4",
          borderRadius: "6px",
        });
      }
    });
  }, [chart]);

  return <div ref={containerRef} style={styles.mermaidContainer} />;
}

function MarkdownCode(props: React.HTMLAttributes<HTMLElement> & { children?: React.ReactNode; className?: string; node?: unknown }) {
  const { children, className, node: _node, ...rest } = props;
  const match = /language-(\w+)/.exec(className || "");
  const lang = match?.[1];
  const code = String(children).replace(/\n$/, "");

  if (lang === "mermaid") {
    return <MermaidBlock chart={code} />;
  }

  // inline code
  if (!match) {
    return <code style={styles.inlineCode} {...rest}>{children}</code>;
  }

  return <code className={className} {...rest}>{children}</code>;
}

/* ── SKILL.md frontmatter helpers ──────────────────────── */

interface SkillMeta {
  name: string;
  description: string;
  license?: string;
  [key: string]: unknown;
}

function parseSkillFrontmatter(content: string): { meta: SkillMeta; body: string } | null {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/);
  if (!match) return null;
  try {
    const doc = yaml.load(match[1]);
    if (!doc || typeof doc !== "object" || !("name" in doc) || !("description" in doc)) return null;
    return { meta: doc as SkillMeta, body: match[2] };
  } catch {
    return null;
  }
}

function SkillFrontmatterCard({ meta }: { meta: SkillMeta }) {
  const extra = Object.entries(meta).filter(([k]) => !["name", "description", "license"].includes(k));
  return (
    <div style={skillStyles.card}>
      <div style={skillStyles.top}>
        <span style={skillStyles.badge}>SKILL</span>
        <span style={skillStyles.name}>{meta.name}</span>
      </div>
      <div style={skillStyles.desc}>{meta.description}</div>
      {meta.license && (
        <div style={skillStyles.field}>
          <span style={skillStyles.fieldLabel}>License</span>
          <span style={skillStyles.fieldValue}>{String(meta.license)}</span>
        </div>
      )}
      {extra.map(([k, v]) => (
        <div key={k} style={skillStyles.field}>
          <span style={skillStyles.fieldLabel}>{k}</span>
          <span style={skillStyles.fieldValue}>{String(v)}</span>
        </div>
      ))}
    </div>
  );
}

const skillStyles: Record<string, React.CSSProperties> = {
  card: {
    margin: "16px 24px 0",
    padding: "16px 20px",
    background: "#FAF8F5",
    borderRadius: 8,
    border: "1px solid #E8E0D4",
  },
  top: {
    display: "flex",
    alignItems: "center",
    gap: 10,
  },
  badge: {
    display: "inline-block",
    padding: "2px 8px",
    borderRadius: 4,
    fontSize: 10,
    fontWeight: 700,
    color: "#fff",
    letterSpacing: "0.05em",
    background: "#6BCB77",
    textTransform: "uppercase" as const,
  },
  name: {
    fontSize: 18,
    fontWeight: 600,
    color: "#2D2A26",
  },
  desc: {
    marginTop: 8,
    fontSize: 13,
    color: "#6B6560",
    lineHeight: 1.5,
  },
  field: {
    display: "flex",
    justifyContent: "space-between",
    marginTop: 6,
    paddingTop: 6,
    borderTop: "1px solid #E8E0D4",
  },
  fieldLabel: {
    fontSize: 12,
    color: "#A09890",
  },
  fieldValue: {
    fontSize: 12,
    color: "#2D2A26",
    fontFamily: "monospace",
  },
};

/* ── File content types ────────────────────────────────── */

interface FileContent {
  path: string;
  content: string;
  size: number;
  truncated: boolean;
  language: string;
}

interface FileViewerProps {
  filePath: string | null;
}

export function FileViewer({ filePath }: FileViewerProps) {
  const [file, setFile] = useState<FileContent | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [yamlMode, setYamlMode] = useState<"raw" | "form">("form");
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!filePath) {
      setFile(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetch(`/files/content?path=${encodeURIComponent(filePath)}`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load file");
        return res.json();
      })
      .then((data: FileContent) => {
        if (!cancelled) setFile(data);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [filePath]);

  // 语法高亮
  useEffect(() => {
    if (file && codeRef.current) {
      codeRef.current.removeAttribute("data-highlighted");
      hljs.highlightElement(codeRef.current);
    }
  }, [file]);

  if (!filePath) {
    return (
      <div style={styles.container}>
        <div style={styles.empty}>Select a file to view its contents</div>
      </div>
    );
  }

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.pathBar}>{filePath}</div>
        <div style={styles.empty}>Loading…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.container}>
        <div style={styles.pathBar}>{filePath}</div>
        <div style={{ ...styles.empty, color: "#FF6B6B" }}>{error}</div>
      </div>
    );
  }

  if (!file) return null;

  const isMarkdown = file.language === "markdown" || file.path.endsWith(".md");
  const isSkillMd = isMarkdown && file.path.endsWith("SKILL.md");
  const skillParsed = isSkillMd ? parseSkillFrontmatter(file.content) : null;

  if (isMarkdown) {
    const markdownContent = skillParsed ? skillParsed.body : file.content;
    return (
      <div style={styles.container}>
        <div style={styles.pathBar}>
          <span style={styles.pathText}>{file.path}</span>
          <span style={styles.meta}>
            {isSkillMd ? "skill" : "markdown"} · {formatSize(file.size)}
            {file.truncated && " · truncated"}
          </span>
        </div>
        <div className="markdown-area" style={styles.markdownArea}>
          {skillParsed && <SkillFrontmatterCard meta={skillParsed.meta} />}
          <div style={skillParsed ? { padding: "0 24px 16px" } : undefined}>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              components={{ code: MarkdownCode as never }}
            >
              {markdownContent}
            </ReactMarkdown>
          </div>
        </div>
      </div>
    );
  }

  const isYaml = file.language === "yaml" || /\.ya?ml$/.test(file.path);
  const yamlConfig = isYaml ? detectYamlConfig(file.content) : null;

  if (isYaml && yamlConfig && yamlMode === "form") {
    return (
      <div style={styles.container}>
        <div style={styles.pathBar}>
          <span style={styles.pathText}>{file.path}</span>
          <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <button onClick={() => setYamlMode("raw")} style={styles.toggleBtn}>
              Raw YAML
            </button>
            <span style={styles.meta}>
              {yamlConfig.configType} config · {formatSize(file.size)}
            </span>
          </span>
        </div>
        <div style={{ flex: 1, overflow: "auto" }}>
          <YamlConfigViewer content={file.content} />
        </div>
      </div>
    );
  }

  const lines = file.content.split("\n");

  return (
    <div style={styles.container}>
      <div style={styles.pathBar}>
        <span style={styles.pathText}>{file.path}</span>
        <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {yamlConfig && (
            <button onClick={() => setYamlMode("form")} style={styles.toggleBtn}>
              Config View
            </button>
          )}
          <span style={styles.meta}>
            {file.language} · {formatSize(file.size)}
            {file.truncated && " · truncated"}
          </span>
        </span>
      </div>
      <div style={styles.codeArea}>
        <div style={styles.lineNumbers}>
          {lines.map((_, i) => (
            <div key={i} style={styles.lineNum}>{i + 1}</div>
          ))}
        </div>
        <pre style={styles.pre}>
          <code ref={codeRef} className={`language-${file.language}`}>
            {file.content}
          </code>
        </pre>
      </div>
    </div>
  );
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100%",
    background: "#FFFFFF",
    borderLeft: "1px solid #E8E0D4",
    borderRight: "1px solid #E8E0D4",
  },
  empty: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#C8C0B8",
    fontSize: 14,
  },
  pathBar: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "8px 14px",
    borderBottom: "1px solid #E8E0D4",
    background: "#FAF8F5",
    flexShrink: 0,
  },
  pathText: {
    fontSize: 12,
    color: "#2D2A26",
    fontFamily: "monospace",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  meta: {
    fontSize: 11,
    color: "#A09890",
    flexShrink: 0,
    marginLeft: 12,
  },
  codeArea: {
    flex: 1,
    display: "flex",
    overflow: "auto",
  },
  lineNumbers: {
    padding: "12px 0",
    textAlign: "right",
    userSelect: "none",
    flexShrink: 0,
    borderRight: "1px solid #E8E0D4",
    background: "#FAF8F5",
  },
  lineNum: {
    padding: "0 10px",
    fontSize: 12,
    lineHeight: "20px",
    color: "#C8C0B8",
    fontFamily: "monospace",
  },
  pre: {
    margin: 0,
    padding: 12,
    flex: 1,
    overflow: "visible",
    fontSize: 13,
    lineHeight: "20px",
    fontFamily: "'Fira Code', 'Cascadia Code', monospace",
    background: "transparent",
  },
  markdownArea: {
    flex: 1,
    overflow: "auto",
    padding: "16px 24px",
    fontSize: 14,
    lineHeight: 1.7,
    color: "#2D2A26",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  },
  mermaidContainer: {
    margin: "16px 0",
    display: "flex",
    justifyContent: "center",
    overflow: "auto",
  },
  inlineCode: {
    background: "#F5F0E8",
    padding: "2px 6px",
    borderRadius: 4,
    fontSize: "0.9em",
    color: "#7C5CBF",
  },
  toggleBtn: {
    background: "#FFFFFF",
    border: "1px solid #E8E0D4",
    borderRadius: 4,
    color: "#6B6560",
    fontSize: 11,
    padding: "3px 10px",
    cursor: "pointer",
    whiteSpace: "nowrap" as const,
  },
};
