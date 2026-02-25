import { useCallback, useRef, useState } from "react";
import { parse as parsePartialJson } from "partial-json";
import { MarkdownRenderer } from "./MarkdownRenderer";

/** 将 LLM 输出中的字面 \n 转为真实换行，以便 Markdown 正确渲染 */
function normalizeNewlines(text: string | undefined): string {
  if (!text) return "";
  return text.replace(/\\n/g, "\n");
}

interface AskUserQuestion {
  question: string;
  header: string;
  type?: "choice" | "text" | "yesno";
  options?: Array<{ label: string; description: string }>;
  multiSelect?: boolean;
  placeholder?: string;
}

interface AskUserBlockProps {
  questions: AskUserQuestion[];
  agentId: string;
  answered: boolean;
  onSendMessage: (content: string, toAgentId: string) => void;
  onSubmitted?: () => void;
}

export function AskUserBlock({
  questions,
  agentId,
  answered,
  onSendMessage,
  onSubmitted,
}: AskUserBlockProps) {
  const [answers, setAnswers] = useState<Record<number, string | string[]>>(
    {}
  );
  const [submitted, setSubmitted] = useState(answered);
  const submittingRef = useRef(false);

  const setAnswer = useCallback((idx: number, value: string | string[]) => {
    setAnswers((prev) => ({ ...prev, [idx]: value }));
  }, []);

  const handleSubmit = useCallback(() => {
    if (submittingRef.current) return;
    const parts: string[] = [];
    for (let i = 0; i < questions.length; i++) {
      const q = questions[i];
      const ans = answers[i];
      const label = q.header || `Q${i + 1}`;
      if (Array.isArray(ans)) {
        parts.push(`${label}: ${ans.join(", ")}`);
      } else if (ans) {
        parts.push(`${label}: ${ans}`);
      }
    }
    if (parts.length === 0) return;
    submittingRef.current = true;
    const message =
      questions.length === 1 ? parts[0] : parts.join("\n");
    onSendMessage(message, agentId);
    setSubmitted(true);
    onSubmitted?.();
  }, [questions, answers, agentId, onSendMessage, onSubmitted]);

  if (submitted) {
    return (
      <div style={{ ...styles.container, opacity: 0.75 }}>
        <div style={styles.headerRow}>
          <span style={styles.badge}>ASK USER</span>
          <span style={styles.answeredLabel}>Answered</span>
        </div>

        {questions.map((q, i) => {
          const qType = q.type || "choice";
          return (
            <div key={i} style={styles.questionBlock}>
              <div style={styles.qHeader}>{q.header}</div>
              <div style={styles.qText}><MarkdownRenderer content={normalizeNewlines(q.question)} /></div>
            {q.multiSelect && (
              <div style={styles.multiSelectHint}>Select multiple</div>
            )}
              {qType === "choice" && q.options && (
                <ChoiceInput
                  options={q.options}
                  multiSelect={q.multiSelect ?? false}
                  value={answers[i]}
                  onChange={() => {}}
                  disabled
                />
              )}
              {qType === "text" && (
                <input
                  type="text"
                  value={(answers[i] as string) || ""}
                  readOnly
                  style={{ ...styles.textInput, cursor: "default" }}
                />
              )}
              {qType === "yesno" && (
                <div style={styles.yesnoRow}>
                  {["Yes", "No"].map((opt) => (
                    <button
                      key={opt}
                      disabled
                      style={{
                        ...styles.optionBtn,
                        ...(answers[i] === opt ? styles.optionBtnSelected : {}),
                        cursor: "default",
                      }}
                    >
                      {opt}
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}

        <div style={styles.submittedLabel}>Submitted</div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.headerRow}>
        <span style={styles.badge}>ASK USER</span>
        <span style={styles.waitingLabel}>Waiting for your answer</span>
      </div>

      {questions.map((q, i) => {
        const qType = q.type || "choice";
        return (
          <div key={i} style={styles.questionBlock}>
            <div style={styles.qHeader}>{q.header}</div>
            <div style={styles.qText}><MarkdownRenderer content={normalizeNewlines(q.question)} /></div>
            {q.multiSelect && (
              <div style={styles.multiSelectHint}>Select multiple</div>
            )}
            {qType === "choice" && q.options && (
              <ChoiceInput
                options={q.options}
                multiSelect={q.multiSelect ?? false}
                value={answers[i]}
                onChange={(v) => setAnswer(i, v)}
              />
            )}
            {qType === "text" && (
              <input
                type="text"
                placeholder={q.placeholder || "Type your answer..."}
                value={(answers[i] as string) || ""}
                onChange={(e) => setAnswer(i, e.target.value)}
                style={styles.textInput}
              />
            )}
            {qType === "yesno" && (
              <div style={styles.yesnoRow}>
                {["Yes", "No"].map((opt) => (
                  <button
                    key={opt}
                    onClick={() => setAnswer(i, opt)}
                    style={{
                      ...styles.optionBtn,
                      ...(answers[i] === opt ? styles.optionBtnSelected : {}),
                    }}
                  >
                    {opt}
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}

      <button onClick={handleSubmit} style={styles.submitBtn}>
        Submit
      </button>
    </div>
  );
}

function ChoiceInput({
  options,
  multiSelect,
  value,
  onChange,
  disabled = false,
}: {
  options: Array<{ label: string; description: string }>;
  multiSelect: boolean;
  value: string | string[] | undefined;
  onChange: (v: string | string[]) => void;
  disabled?: boolean;
}) {
  const [otherActive, setOtherActive] = useState(false);
  const [otherText, setOtherText] = useState("");

  const selected = multiSelect
    ? Array.isArray(value)
      ? value
      : []
    : typeof value === "string"
      ? value
      : "";

  const optionLabels = options.map((o) => o.label);

  // For single-select: check if current value is a custom "other" answer
  const isOtherValue =
    !multiSelect &&
    typeof selected === "string" &&
    selected !== "" &&
    !optionLabels.includes(selected);

  const effectiveOtherActive = otherActive || isOtherValue;

  const toggle = (label: string) => {
    if (disabled) return;
    if (multiSelect) {
      const arr = selected as string[];
      const filtered = arr.filter((l) => optionLabels.includes(l));
      const next = filtered.includes(label)
        ? filtered.filter((l) => l !== label)
        : [...filtered, label];
      // Keep "other" text if active
      if (otherActive && otherText) {
        onChange([...next, `Other: ${otherText}`]);
      } else {
        onChange(next);
      }
    } else {
      setOtherActive(false);
      onChange(label);
    }
  };

  const handleOtherClick = () => {
    if (disabled) return;
    if (multiSelect) {
      const newActive = !otherActive;
      setOtherActive(newActive);
      const arr = (selected as string[]).filter((l) => optionLabels.includes(l));
      if (newActive && otherText) {
        onChange([...arr, `Other: ${otherText}`]);
      } else {
        onChange(arr);
      }
    } else {
      setOtherActive(true);
      onChange(otherText || "");
    }
  };

  const handleOtherTextChange = (text: string) => {
    setOtherText(text);
    if (multiSelect) {
      const arr = (selected as string[]).filter((l) => optionLabels.includes(l));
      if (text) {
        onChange([...arr, `Other: ${text}`]);
      } else {
        onChange(arr);
      }
    } else {
      onChange(text);
    }
  };

  return (
    <div style={styles.optionsGrid}>
      {options.map((opt) => {
        const isSelected = multiSelect
          ? (selected as string[]).includes(opt.label)
          : selected === opt.label;
        return (
          <button
            key={opt.label}
            onClick={() => toggle(opt.label)}
            disabled={disabled}
            style={{
              ...styles.optionBtn,
              ...(isSelected ? styles.optionBtnSelected : {}),
              ...(disabled ? { cursor: "default" } : {}),
            }}
          >
            <span style={styles.optLabel}>{opt.label}</span>
            {opt.description && (
              <span style={styles.optDesc}>{opt.description}</span>
            )}
          </button>
        );
      })}

      {/* "Other" option */}
      <button
        onClick={handleOtherClick}
        disabled={disabled}
        style={{
          ...styles.optionBtn,
          ...(effectiveOtherActive ? styles.optionBtnSelected : {}),
          ...(disabled ? { cursor: "default" } : {}),
        }}
      >
        <span style={styles.optLabel}>Other</span>
        <span style={styles.optDesc}>Provide your own answer</span>
      </button>

      {effectiveOtherActive && (
        <input
          type="text"
          autoFocus={!disabled}
          placeholder="Type your answer..."
          value={otherText || (isOtherValue ? (selected as string) : "")}
          onChange={(e) => handleOtherTextChange(e.target.value)}
          readOnly={disabled}
          style={{ ...styles.textInput, ...(disabled ? { cursor: "default" } : {}) }}
        />
      )}
    </div>
  );
}

/** Try to parse ask_user questions from tool call args JSON.
 *  支持 partial-json 解析流式参数。
 */
export function parseAskUserQuestions(
  argsJson: string,
  partial = false
): AskUserQuestion[] | null {
  try {
    const parsed = partial ? parsePartialJson(argsJson) : JSON.parse(argsJson);
    if (
      parsed &&
      typeof parsed === "object" &&
      Array.isArray(parsed.questions)
    ) {
      return parsed.questions as AskUserQuestion[];
    }
  } catch {
    // args not yet complete
  }
  return null;
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    margin: "8px 0",
    padding: "12px 14px",
    borderRadius: 8,
    background: "#F0E6FF",
    border: "1px solid #D4C4F0",
    display: "flex",
    flexDirection: "column",
    gap: 12,
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
    background: "#FFFFFF",
    border: "1px solid #D4C4F0",
    color: "#7C5CBF",
  },
  waitingLabel: {
    fontSize: 12,
    color: "#7C5CBF",
    fontWeight: 500,
  },
  answeredLabel: {
    fontSize: 12,
    color: "#6BCB77",
    fontWeight: 500,
  },
  questionBlock: {
    display: "flex",
    flexDirection: "column",
    gap: 6,
  },
  qHeader: {
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: 0.4,
    textTransform: "uppercase",
    color: "#7C5CBF",
  },
  qText: {
    fontSize: 13,
    color: "#2D2A26",
    lineHeight: 1.4,
  },
  qAnswer: {
    fontSize: 13,
    color: "#2D2A26",
    marginLeft: 8,
  },
  answeredQ: {
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  optionsGrid: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
    marginTop: 4,
  },
  optionBtn: {
    display: "flex",
    flexDirection: "column",
    alignItems: "flex-start",
    gap: 2,
    padding: "8px 12px",
    borderRadius: 6,
    border: "1px solid #E8E0D4",
    background: "#FFFFFF",
    color: "#2D2A26",
    fontSize: 13,
    cursor: "pointer",
    textAlign: "left",
  },
  optionBtnSelected: {
    borderColor: "#A78BFA",
    background: "#F0E6FF",
  },
  optLabel: {
    fontWeight: 600,
    fontSize: 13,
  },
  optDesc: {
    fontSize: 11,
    color: "#A09890",
  },
  yesnoRow: {
    display: "flex",
    gap: 8,
    marginTop: 4,
  },
  textInput: {
    padding: "8px 12px",
    borderRadius: 6,
    border: "1px solid #E8E0D4",
    background: "#FFFFFF",
    color: "#2D2A26",
    fontSize: 13,
    outline: "none",
    marginTop: 4,
  },
  submitBtn: {
    alignSelf: "flex-end",
    padding: "8px 20px",
    borderRadius: 6,
    border: "none",
    background: "#A78BFA",
    color: "#FFFFFF",
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
  },
  submittedLabel: {
    alignSelf: "flex-end",
    fontSize: 12,
    color: "#6BCB77",
    fontWeight: 500,
  },
  multiSelectHint: {
    fontSize: 11,
    color: "#7C5CBF",
    fontStyle: "italic",
  },
};
