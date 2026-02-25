import { useCallback, useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import mermaid from "mermaid";
import "highlight.js/styles/github.css";

// 1. 初始化 mermaid，使用暗色主题
mermaid.initialize({
  startOnLoad: false,
  theme: "default",
  securityLevel: "loose",
});

// 缩放工具栏（内联 + 全屏共用）
function ZoomBar({
  zoom,
  onZoomIn,
  onZoomOut,
  onReset,
  extra,
}: {
  zoom: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
  extra?: React.ReactNode;
}) {
  return (
    <div style={mdStyles.zoomBar}>
      <button style={mdStyles.zoomBtn} onClick={onZoomOut}>−</button>
      <span style={mdStyles.zoomLabel}>{Math.round(zoom * 100)}%</span>
      <button style={mdStyles.zoomBtn} onClick={onZoomIn}>+</button>
      <button
        style={{ ...mdStyles.zoomBtn, marginLeft: 4, fontSize: 11 }}
        onClick={onReset}
      >
        Reset
      </button>
      {extra}
    </div>
  );
}

// 从 SVG 字符串中提取原始宽度（mermaid 生成的 SVG 带有 width/viewBox 属性）
function parseSvgNaturalWidth(svgString: string): number {
  const tmp = document.createElement("div");
  tmp.innerHTML = svgString;
  const svgEl = tmp.querySelector("svg");
  if (!svgEl) return 0;

  // 优先从 width 属性取（mermaid 通常设置 width="1234"）
  const w = svgEl.getAttribute("width");
  if (w) {
    const n = parseFloat(w);
    if (n > 0) return n;
  }

  // 其次从 viewBox 取第三个值
  const vb = svgEl.getAttribute("viewBox");
  if (vb) {
    const parts = vb.trim().split(/[\s,]+/);
    if (parts.length >= 3) {
      const n = parseFloat(parts[2]);
      if (n > 0) return n;
    }
  }

  return 0;
}

// 全屏预览弹窗
function MermaidFullscreen({
  svg,
  onClose,
}: {
  svg: string;
  onClose: () => void;
}) {
  const [zoom, setZoom] = useState(1);
  const [fitZoom, setFitZoom] = useState(1);
  const scrollRef = useRef<HTMLDivElement>(null);

  // 计算 fit-width * 80% 的缩放比例
  useEffect(() => {
    const scrollEl = scrollRef.current;
    if (!scrollEl) return;
    const containerWidth = scrollEl.clientWidth - 40; // 减去 padding
    const svgNaturalWidth = parseSvgNaturalWidth(svg);
    if (svgNaturalWidth > 0 && containerWidth > 0) {
      const fit = (containerWidth * 0.8) / svgNaturalWidth;
      const clamped = Math.min(3, Math.max(0.1, fit));
      setFitZoom(clamped);
      setZoom(clamped);
    }
  }, [svg]);

  // ESC 关闭
  const handleKey = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    },
    [onClose]
  );

  useEffect(() => {
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [handleKey]);

  return createPortal(
    <div style={mdStyles.overlay} onClick={onClose}>
      <div style={mdStyles.fullscreenCard} onClick={(e) => e.stopPropagation()}>
        <ZoomBar
          zoom={zoom}
          onZoomIn={() => setZoom((z) => Math.min(3, z + 0.1))}
          onZoomOut={() => setZoom((z) => Math.max(0.1, z - 0.1))}
          onReset={() => setZoom(fitZoom)}
          extra={
            <button
              style={{ ...mdStyles.zoomBtn, marginLeft: "auto", fontSize: 11 }}
              onClick={onClose}
            >
              Close (Esc)
            </button>
          }
        />
        <div ref={scrollRef} style={mdStyles.fullscreenScroll}>
          <div style={mdStyles.fullscreenCenter}>
            <div
              style={{
                ...mdStyles.mermaidInner,
                transform: `scale(${zoom})`,
              }}
              dangerouslySetInnerHTML={{ __html: svg }}
            />
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}

function MermaidBlock({ code }: { code: string }) {
  const id = useId().replace(/:/g, "_");
  const [svg, setSvg] = useState<string>("");
  const [zoom, setZoom] = useState(0.7);
  const [fullscreen, setFullscreen] = useState(false);
  const renderCount = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout>>(null);

  // debounce 渲染：流式输出时 code 持续变化，等稳定后再渲染
  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);

    timerRef.current = setTimeout(() => {
      const currentRender = ++renderCount.current;
      (async () => {
        try {
          const { svg: rendered } = await mermaid.render(
            `mermaid_${id}_${currentRender}`,
            code.trim()
          );
          if (currentRender === renderCount.current) setSvg(rendered);
        } catch {
          // mermaid v11 会在 DOM 中残留错误元素，清理掉
          const errEl = document.getElementById(`mermaid_${id}_${currentRender}`);
          if (errEl) errEl.remove();
          // 渲染失败时保留上一次成功的 SVG，避免闪烁
        }
      })();
    }, 800);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [code, id]);

  // 有成功渲染的 SVG 则展示可缩放图表，否则显示原始代码
  if (svg) {
    return (
      <>
        <div style={mdStyles.mermaidOuter}>
          <ZoomBar
            zoom={zoom}
            onZoomIn={() => setZoom((z) => Math.min(2, z + 0.1))}
            onZoomOut={() => setZoom((z) => Math.max(0.2, z - 0.1))}
            onReset={() => setZoom(0.7)}
            extra={
              <button
                style={{ ...mdStyles.zoomBtn, marginLeft: "auto", fontSize: 11 }}
                onClick={() => setFullscreen(true)}
              >
                Fullscreen
              </button>
            }
          />
          <div style={mdStyles.mermaidScroll}>
            <div
              style={{
                ...mdStyles.mermaidInner,
                transform: `scale(${zoom})`,
              }}
              dangerouslySetInnerHTML={{ __html: svg }}
            />
          </div>
        </div>
        {fullscreen && (
          <MermaidFullscreen svg={svg} onClose={() => setFullscreen(false)} />
        )}
      </>
    );
  }

  return (
    <pre style={mdStyles.codeBlock}>
      <code>{code}</code>
    </pre>
  );
}

// 从 React children 中递归提取纯文本（用于 mermaid 和 inline 判断）
function extractText(node: React.ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string") return node;
  if (typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (typeof node === "object" && "props" in node) {
    const props = (node as React.ReactElement).props as { children?: React.ReactNode };
    return extractText(props.children);
  }
  return "";
}

// 2. 自定义 code block 渲染器
function CodeBlock({
  className,
  children,
}: {
  className?: string;
  children?: React.ReactNode;
}) {
  const langMatch = className?.match(/language-(\w+)/);
  const lang = langMatch?.[1] ?? "";

  // mermaid 特殊处理：需要纯文本
  if (lang === "mermaid") {
    const code = extractText(children).replace(/\n$/, "");
    return <MermaidBlock code={code} />;
  }

  // 普通代码块：直接渲染 children，保留 rehype-highlight 的语法高亮
  return (
    <pre style={mdStyles.codeBlock}>
      <code className={className}>{children}</code>
    </pre>
  );
}

// 3. 内联 code 渲染器
function InlineCode({ children }: { children?: React.ReactNode }) {
  return <code style={mdStyles.inlineCode}>{children}</code>;
}

// 4. 主组件
export function MarkdownRenderer({ content }: { content: string }) {
  return (
    <div style={mdStyles.root}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          pre({ children }) {
            return <>{children}</>;
          },
          code({ className, children }) {
            const isInline = !className && !extractText(children).includes("\n");
            if (isInline) {
              return <InlineCode>{children}</InlineCode>;
            }
            return (
              <CodeBlock className={className}>
                {children}
              </CodeBlock>
            );
          },
          table({ children }) {
            return (
              <div style={mdStyles.tableWrapper}>
                <table style={mdStyles.table}>
                  {children}
                </table>
              </div>
            );
          },
          th({ children }) {
            return <th style={mdStyles.th}>{children}</th>;
          },
          td({ children }) {
            return <td style={mdStyles.td}>{children}</td>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

const mdStyles: Record<string, React.CSSProperties> = {
  root: {
    wordBreak: "break-word",
    lineHeight: 1.6,
  },
  codeBlock: {
    margin: "8px 0",
    padding: "10px 12px",
    fontSize: 12,
    lineHeight: 1.5,
    color: "#2D2A26",
    background: "#F5F0E8",
    border: "1px solid #E8E0D4",
    borderRadius: 6,
    overflowX: "auto",
  },
  inlineCode: {
    padding: "1px 5px",
    fontSize: "0.9em",
    background: "#F5F0E8",
    border: "1px solid #E8E0D4",
    borderRadius: 4,
    color: "#7C5CBF",
  },
  tableWrapper: {
    overflowX: "auto",
    margin: "8px 0",
  },
  table: {
    borderCollapse: "collapse",
    width: "100%",
    fontSize: 12,
  },
  th: {
    padding: "6px 12px",
    borderBottom: "2px solid #E8E0D4",
    background: "#FAF8F5",
    color: "#2D2A26",
    fontWeight: 600,
    textAlign: "left",
  },
  td: {
    padding: "6px 12px",
    borderBottom: "1px solid #E8E0D4",
    color: "#2D2A26",
  },
  mermaidOuter: {
    margin: "8px 0",
    background: "#FFFFFF",
    border: "1px solid #E8E0D4",
    borderRadius: 6,
    overflow: "hidden",
  },
  zoomBar: {
    display: "flex",
    alignItems: "center",
    gap: 4,
    padding: "4px 8px",
    borderBottom: "1px solid #E8E0D4",
    background: "#FAF8F5",
  },
  zoomBtn: {
    background: "#FFFFFF",
    border: "1px solid #E8E0D4",
    borderRadius: 4,
    color: "#6B6560",
    cursor: "pointer",
    fontSize: 14,
    lineHeight: 1,
    padding: "2px 8px",
  },
  zoomLabel: {
    fontSize: 11,
    color: "#A09890",
    minWidth: 36,
    textAlign: "center" as const,
  },
  mermaidScroll: {
    overflow: "auto",
    maxHeight: 480,
    padding: 12,
  },
  mermaidInner: {
    transformOrigin: "top left",
    display: "inline-block",
  },
  overlay: {
    position: "fixed" as const,
    inset: 0,
    zIndex: 9999,
    background: "rgba(0, 0, 0, 0.5)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  fullscreenCard: {
    width: "95vw",
    height: "92vh",
    background: "#FFFFFF",
    border: "1px solid #E8E0D4",
    borderRadius: 8,
    display: "flex",
    flexDirection: "column" as const,
    overflow: "hidden",
  },
  fullscreenScroll: {
    flex: 1,
    overflow: "auto",
    padding: 20,
  },
  fullscreenCenter: {
    display: "flex",
    justifyContent: "left",
    minHeight: "100%",
  },
  mermaidLoading: {
    padding: 12,
    color: "#A09890",
    fontSize: 12,
    fontStyle: "italic",
  },
  errorBlock: {
    margin: "8px 0",
    padding: "10px 12px",
    fontSize: 12,
    color: "#2D2A26",
    background: "#FFF0F0",
    border: "1px solid #FFD0D0",
    borderRadius: 6,
    overflowX: "auto",
  },
};
