import { useCallback, useRef, useState } from "react";
import {
  AlertTriangle, ChevronLeft, ChevronRight, Check, Minimize2,
} from "lucide-react";
import type { AskUserQuestion } from "./AskUserBlock";
import { MarkdownRenderer } from "./MarkdownRenderer";

function normalizeNewlines(text: string | undefined): string {
  if (!text) return "";
  return text.replace(/\\n/g, "\n");
}

interface AskUserModalProps {
  questions: AskUserQuestion[];
  agentId: string;
  roleName?: string;
  onSendMessage: (content: string, toAgentId: string) => void;
  onSubmitted: () => void;
  onHide?: () => void;
}

export function AskUserModal({
  questions, agentId, roleName, onSendMessage, onSubmitted, onHide,
}: AskUserModalProps) {
  const [page, setPage] = useState(0);
  const [answers, setAnswers] = useState<Record<number, string | string[]>>({});
  const submittingRef = useRef(false);

  const total = questions.length;
  const q = questions[page];

  const setAnswer = useCallback((idx: number, value: string | string[]) => {
    setAnswers((prev) => ({ ...prev, [idx]: value }));
  }, []);

  const allAnswered = questions.every((_, i) => {
    const a = answers[i];
    if (Array.isArray(a)) return a.length > 0;
    return typeof a === "string" && a.length > 0;
  });

  const handleSubmit = useCallback(() => {
    if (submittingRef.current || !allAnswered) return;
    submittingRef.current = true;
    const parts: string[] = [];
    for (let i = 0; i < questions.length; i++) {
      const label = questions[i].header || `Q${i + 1}`;
      const ans = answers[i];
      if (Array.isArray(ans)) parts.push(`${label}: ${ans.join(", ")}`);
      else if (ans) parts.push(`${label}: ${ans}`);
    }
    const message = questions.length === 1 ? parts[0] : parts.join("\n");
    onSendMessage(message, agentId);
    onSubmitted();
  }, [allAnswered, questions, answers, agentId, onSendMessage, onSubmitted]);

  const qType = q?.type || "choice";

  return (
    <div className="fixed inset-0 bg-black/20 z-50 flex items-center justify-center">
      <div className="bg-white border border-amber-300 shadow-2xl w-[460px] animate-in zoom-in-95">
        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-5 pb-3 border-b border-amber-100">
          <div className="flex items-center gap-2 text-amber-600">
            <AlertTriangle size={16} className="animate-pulse shrink-0" />
            <h3 className="text-xs font-bold tracking-widest uppercase">Human Intervention Required</h3>
          </div>
          {onHide && (
            <button
              onClick={onHide}
              className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
              title="Minimize"
            >
              <Minimize2 size={14} />
            </button>
          )}
        </div>

        {/* Source */}
        <div className="px-6 pt-3 text-[10px] font-mono text-slate-500">
          Source: <span className="text-blue-600 font-bold">{roleName || agentId}</span>
        </div>

        {/* Question area */}
        <div className="px-6 py-4 min-h-[180px]">
          {/* Question header + text */}
          <div className="mb-1 text-[11px] font-bold uppercase tracking-wider text-amber-700">
            {q.header}
          </div>
          <div className="text-sm text-slate-800 leading-relaxed mb-4">
            <MarkdownRenderer content={normalizeNewlines(q.question)} />
          </div>
          {q.multiSelect && (
            <div className="text-[11px] text-amber-600 italic mb-2">Select multiple</div>
          )}

          {/* Choice */}
          {qType === "choice" && q.options && (
            <ChoiceInput
              key={page}
              options={q.options}
              multiSelect={q.multiSelect ?? false}
              value={answers[page]}
              onChange={(v) => setAnswer(page, v)}
            />
          )}

          {/* Text */}
          {qType === "text" && (
            <input
              type="text"
              autoFocus
              placeholder={q.placeholder || "Type your answer..."}
              value={(answers[page] as string) || ""}
              onChange={(e) => setAnswer(page, e.target.value)}
              className="w-full text-sm px-3 py-2.5 border border-slate-200 bg-slate-50 outline-none focus:border-amber-400 placeholder:text-slate-400"
            />
          )}

          {/* Yes/No */}
          {qType === "yesno" && (
            <div className="flex gap-3">
              {["Yes", "No"].map((opt) => (
                <button
                  key={opt}
                  onClick={() => setAnswer(page, opt)}
                  className={`flex-1 px-4 py-2.5 text-sm font-medium border transition-all ${
                    answers[page] === opt
                      ? "border-amber-400 bg-amber-50 text-amber-700"
                      : "border-slate-200 bg-white text-slate-600 hover:border-amber-300 hover:bg-amber-50/50"
                  }`}
                >
                  {opt}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Footer: navigation + submit */}
        <div className="px-6 pb-5 flex items-center justify-between">
          {/* Left: page nav */}
          <div className="flex items-center gap-2">
            {total > 1 && (
              <>
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="p-1.5 border border-slate-200 text-slate-500 hover:text-slate-700 hover:border-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeft size={14} />
                </button>
                {/* Dots */}
                <div className="flex items-center gap-1.5">
                  {questions.map((_, i) => {
                    const answered = Array.isArray(answers[i])
                      ? (answers[i] as string[]).length > 0
                      : typeof answers[i] === "string" && (answers[i] as string).length > 0;
                    return (
                      <button
                        key={i}
                        onClick={() => setPage(i)}
                        className={`w-2 h-2 rounded-full transition-all ${
                          i === page
                            ? "bg-amber-500 scale-125"
                            : answered
                              ? "bg-emerald-400"
                              : "bg-slate-300"
                        }`}
                      />
                    );
                  })}
                </div>
                <button
                  onClick={() => setPage((p) => Math.min(total - 1, p + 1))}
                  disabled={page === total - 1}
                  className="p-1.5 border border-slate-200 text-slate-500 hover:text-slate-700 hover:border-slate-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronRight size={14} />
                </button>
                <span className="text-[10px] text-slate-400 font-mono ml-1">{page + 1}/{total}</span>
              </>
            )}
          </div>

          {/* Right: submit */}
          <button
            onClick={handleSubmit}
            disabled={!allAnswered}
            className="flex items-center gap-1.5 px-5 py-2 text-xs font-bold uppercase tracking-wider border transition-all disabled:opacity-40 disabled:cursor-not-allowed border-amber-400 bg-amber-500 text-white hover:bg-amber-600"
          >
            <Check size={12} strokeWidth={3} />
            Submit
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChoiceInput (Tailwind version for modal)
// ---------------------------------------------------------------------------
function ChoiceInput({
  options, multiSelect, value, onChange,
}: {
  options: Array<{ label: string; description: string }>;
  multiSelect: boolean;
  value: string | string[] | undefined;
  onChange: (v: string | string[]) => void;
}) {
  const [otherActive, setOtherActive] = useState(false);
  const [otherText, setOtherText] = useState("");

  const selected = multiSelect
    ? Array.isArray(value) ? value : []
    : typeof value === "string" ? value : "";

  const optionLabels = options.map((o) => o.label);
  const isOtherValue =
    !multiSelect && typeof selected === "string" && selected !== "" && !optionLabels.includes(selected);
  const effectiveOtherActive = otherActive || isOtherValue;

  const toggle = (label: string) => {
    if (multiSelect) {
      const arr = (selected as string[]).filter((l) => optionLabels.includes(l));
      const next = arr.includes(label) ? arr.filter((l) => l !== label) : [...arr, label];
      if (otherActive && otherText) onChange([...next, `Other: ${otherText}`]);
      else onChange(next);
    } else {
      setOtherActive(false);
      onChange(label);
    }
  };

  const handleOtherClick = () => {
    if (multiSelect) {
      const newActive = !otherActive;
      setOtherActive(newActive);
      const arr = (selected as string[]).filter((l) => optionLabels.includes(l));
      if (newActive && otherText) onChange([...arr, `Other: ${otherText}`]);
      else onChange(arr);
    } else {
      setOtherActive(true);
      onChange(otherText || "");
    }
  };

  const handleOtherTextChange = (text: string) => {
    setOtherText(text);
    if (multiSelect) {
      const arr = (selected as string[]).filter((l) => optionLabels.includes(l));
      if (text) onChange([...arr, `Other: ${text}`]);
      else onChange(arr);
    } else {
      onChange(text);
    }
  };

  return (
    <div className="flex flex-col gap-1.5">
      {options.map((opt) => {
        const isSelected = multiSelect
          ? (selected as string[]).includes(opt.label)
          : selected === opt.label;
        return (
          <button
            key={opt.label}
            onClick={() => toggle(opt.label)}
            className={`w-full text-left px-4 py-2.5 border text-sm transition-all group flex justify-between items-center ${
              isSelected
                ? "border-amber-400 bg-amber-50 text-amber-800"
                : "border-slate-200 bg-white text-slate-700 hover:border-amber-300 hover:bg-slate-50"
            }`}
          >
            <div className="flex flex-col gap-0.5">
              <span className="font-medium"><MarkdownRenderer content={normalizeNewlines(opt.label)} inline /></span>
              {opt.description && (
                <span className="text-[11px] text-slate-400"><MarkdownRenderer content={normalizeNewlines(opt.description)} inline /></span>
              )}
            </div>
            {isSelected && <Check size={14} className="text-amber-600 shrink-0" />}
          </button>
        );
      })}

      {/* Other option */}
      <button
        onClick={handleOtherClick}
        className={`w-full text-left px-4 py-2.5 border text-sm transition-all ${
          effectiveOtherActive
            ? "border-amber-400 bg-amber-50 text-amber-800"
            : "border-slate-200 bg-white text-slate-700 hover:border-amber-300 hover:bg-slate-50"
        }`}
      >
        <span className="font-medium">Other</span>
        <span className="text-[11px] text-slate-400 ml-2">Provide your own answer</span>
      </button>

      {effectiveOtherActive && (
        <input
          type="text"
          autoFocus
          placeholder="Type your answer..."
          value={otherText || (isOtherValue ? (selected as string) : "")}
          onChange={(e) => handleOtherTextChange(e.target.value)}
          className="w-full text-sm px-3 py-2.5 border border-slate-200 bg-slate-50 outline-none focus:border-amber-400 placeholder:text-slate-400"
        />
      )}
    </div>
  );
}
