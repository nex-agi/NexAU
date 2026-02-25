import { useRef, useState, useCallback } from "react";

interface ChatInputProps {
  onSend: (message: string, toAgentId?: string) => void;
  isStreaming: boolean;
  /** Active agent IDs for the target selector during streaming. */
  agentIds: string[];
}

export function ChatInput({ onSend, isStreaming, agentIds }: ChatInputProps) {
  const [input, setInput] = useState("");
  const [targetAgent, setTargetAgent] = useState("leader");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const composingRef = useRef(false);

  const resetHeight = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed) return;
    onSend(trimmed, isStreaming ? targetAgent : undefined);
    setInput("");
    // reset textarea height after send
    requestAnimationFrame(() => {
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
      }
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // IME 输入法选字时 Enter 不应触发发送
    if (e.key === "Enter" && !e.shiftKey && !composingRef.current) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      {isStreaming && agentIds.length > 0 && (
        <select
          value={targetAgent}
          onChange={(e) => setTargetAgent(e.target.value)}
          style={styles.select}
        >
          {agentIds.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
      )}
      <textarea
        ref={textareaRef}
        value={input}
        onChange={(e) => {
          setInput(e.target.value);
          resetHeight();
        }}
        onKeyDown={handleKeyDown}
        onCompositionStart={() => { composingRef.current = true; }}
        onCompositionEnd={() => { composingRef.current = false; }}
        placeholder={
          isStreaming
            ? `Message to ${targetAgent}...`
            : "Enter a task for the team..."
        }
        rows={1}
        style={styles.textarea}
      />
      <button type="submit" disabled={!input.trim()} style={styles.button}>
        Send
      </button>
    </form>
  );
}

const styles: Record<string, React.CSSProperties> = {
  form: {
    display: "flex",
    gap: 8,
    padding: 16,
    borderTop: "1px solid #E8E0D4",
    background: "#FFFFFF",
    alignItems: "flex-end",
  },
  select: {
    padding: "10px 8px",
    borderRadius: 8,
    border: "1px solid #E8E0D4",
    background: "#FAF8F5",
    color: "#2D2A26",
    fontSize: 13,
    outline: "none",
    minWidth: 120,
  },
  textarea: {
    flex: 1,
    padding: "10px 14px",
    borderRadius: 8,
    border: "1px solid #E8E0D4",
    background: "#FAF8F5",
    color: "#2D2A26",
    fontSize: 14,
    outline: "none",
    resize: "none" as const,
    lineHeight: 1.5,
    fontFamily: "inherit",
    maxHeight: 200,
    overflow: "auto",
  },
  button: {
    padding: "10px 20px",
    borderRadius: 8,
    border: "none",
    background: "#4ECDC4",
    color: "#FFFFFF",
    fontSize: 14,
    fontWeight: 600,
    cursor: "pointer",
  },
};
