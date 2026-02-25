import { useState, useEffect, useCallback, useRef } from "react";

const ChevronDown = () => (
  <svg width="10" height="10" viewBox="0 0 10 10" fill="none" style={{ flexShrink: 0 }}>
    <path d="M2 3.5L5 6.5L8 3.5" stroke="#A09890" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const ChevronRight = () => (
  <svg width="10" height="10" viewBox="0 0 10 10" fill="none" style={{ flexShrink: 0 }}>
    <path d="M3.5 2L6.5 5L3.5 8" stroke="#A09890" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const FolderOpenIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
    <path d="M2 4C2 3.44772 2.44772 3 3 3H6.5L8 5H13C13.5523 5 14 5.44772 14 6V6H3.5L2 12V4Z" fill="#F5C542"/>
    <path d="M2 12L3.5 6H14.5L13 12H2Z" fill="#F7D06B"/>
  </svg>
);

const FolderClosedIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
    <path d="M2 4C2 3.44772 2.44772 3 3 3H6.5L8 5H13C13.5523 5 14 5.44772 14 6V12C14 12.5523 13.5523 13 13 13H3C2.44772 13 2 12.5523 2 12V4Z" fill="#F5C542"/>
    <path d="M2 6H14V12C14 12.5523 13.5523 13 13 13H3C2.44772 13 2 12.5523 2 12V6Z" fill="#F7D06B"/>
  </svg>
);

const FileIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
    <path d="M4 2C3.44772 2 3 2.44772 3 3V13C3 13.5523 3.44772 14 4 14H12C12.5523 14 13 13.5523 13 13V6L9 2H4Z" fill="#E8E0D4"/>
    <path d="M9 2V5C9 5.55228 9.44772 6 10 6H13L9 2Z" fill="#C8C0B8"/>
  </svg>
);

interface FileNode {
  name: string;
  path: string;
  isDirectory: boolean;
  isFile: boolean;
  size: number;
  children?: FileNode[];
  expanded?: boolean;
}

interface FileTreeProps {
  selectedPath: string | null;
  onSelectFile: (path: string) => void;
  isStreaming: boolean;
}

async function fetchTree(path: string): Promise<FileNode[]> {
  const res = await fetch(`/files/tree?path=${encodeURIComponent(path)}`);
  if (!res.ok) return [];
  return res.json();
}

function TreeNode({
  node, depth, selectedPath, onSelectFile, onToggle,
}: {
  node: FileNode;
  depth: number;
  selectedPath: string | null;
  onSelectFile: (path: string) => void;
  onToggle: (node: FileNode) => void;
}) {
  const isSelected = node.path === selectedPath;
  const pl = 8 + depth * 14;

  return (
    <>
      <div
        style={{
          ...ns.row,
          paddingLeft: pl,
          background: isSelected ? "#F0E6FF" : "transparent",
        }}
        onClick={() =>
          node.isDirectory ? onToggle(node) : onSelectFile(node.path)
        }
      >
        <span style={ns.icon}>
          {node.isDirectory ? (
            <>
              {node.expanded ? <ChevronDown /> : <ChevronRight />}
              <span style={{ marginLeft: 2 }}>{node.expanded ? <FolderOpenIcon /> : <FolderClosedIcon />}</span>
            </>
          ) : (
            <span style={{ marginLeft: 12 }}><FileIcon /></span>
          )}
        </span>
        <span
          style={{
            ...ns.name,
            color: isSelected
              ? "#2D2A26"
              : node.isDirectory
                ? "#6B6560"
                : "#A09890",
          }}
        >
          {node.name}
        </span>
      </div>
      {node.expanded &&
        node.children?.map((c) => (
          <TreeNode
            key={c.path}
            node={c}
            depth={depth + 1}
            selectedPath={selectedPath}
            onSelectFile={onSelectFile}
            onToggle={onToggle}
          />
        ))}
    </>
  );
}

export function FileTree({
  selectedPath,
  onSelectFile,
  isStreaming,
}: FileTreeProps) {
  const [roots, setRoots] = useState<FileNode[]>([]);
  const nodesRef = useRef<Map<string, FileNode>>(new Map());

  const loadDir = useCallback(
    async (dirPath: string): Promise<FileNode[]> => {
      const items = await fetchTree(dirPath);
      return items.map((item) => {
        const existing = nodesRef.current.get(item.path);
        const node: FileNode = {
          ...item,
          expanded: existing?.expanded ?? false,
          children: existing?.children,
        };
        nodesRef.current.set(item.path, node);
        return node;
      });
    },
    [],
  );

  const refresh = useCallback(async () => {
    const newRoots = await loadDir(".");
    const refreshExpanded = async (
      nodes: FileNode[],
    ): Promise<FileNode[]> => {
      for (const n of nodes) {
        if (n.isDirectory && n.expanded) {
          const children = await loadDir(n.path);
          n.children = await refreshExpanded(children);
        }
      }
      return nodes;
    };
    setRoots(await refreshExpanded(newRoots));
  }, [loadDir]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (!isStreaming) return;
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [isStreaming, refresh]);

  const handleToggle = useCallback(
    async (node: FileNode) => {
      node.expanded = !node.expanded;
      if (node.expanded && !node.children) {
        node.children = await loadDir(node.path);
      }
      setRoots((prev) => [...prev]);
    },
    [loadDir],
  );

  return (
    <div style={styles.container}>
      <div style={styles.header}>Files</div>
      <div style={styles.tree}>
        {roots.length === 0 ? (
          <div style={styles.empty}>No files yet</div>
        ) : (
          roots.map((n) => (
            <TreeNode
              key={n.path}
              node={n}
              depth={0}
              selectedPath={selectedPath}
              onSelectFile={onSelectFile}
              onToggle={handleToggle}
            />
          ))
        )}
      </div>
    </div>
  );
}

const ns: Record<string, React.CSSProperties> = {
  row: {
    display: "flex",
    alignItems: "center",
    gap: 4,
    padding: "3px 6px",
    cursor: "pointer",
    fontSize: 12,
    whiteSpace: "nowrap",
    overflow: "hidden",
    borderRadius: 3,
  },
  icon: { display: "flex", alignItems: "center", flexShrink: 0, width: 32 },
  name: { overflow: "hidden", textOverflow: "ellipsis" },
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100%",
    background: "#FFFFFF",
    borderLeft: "1px solid #E8E0D4",
  },
  header: {
    padding: "10px 12px",
    fontSize: 12,
    fontWeight: 600,
    color: "#A09890",
    textTransform: "uppercase",
    letterSpacing: 0.5,
    borderBottom: "1px solid #E8E0D4",
    flexShrink: 0,
  },
  tree: { flex: 1, overflow: "auto", padding: "4px 0" },
  empty: {
    padding: 12,
    color: "#C8C0B8",
    fontSize: 12,
    textAlign: "center",
  },
};
