"""
AskUser middleware for NexAU.

Intercepts ask_user tool calls, displays questions to the user, waits for input,
then injects user_answers and was_cancelled into the tool parameters.

Default: uses CLI provider (stdout/stdin). Override by passing user_input_provider
or setting global_storage["ask_user_input_provider"] for frontend integration.
"""

from collections.abc import Callable
from typing import Any, cast

from nexau.archs.main_sub.execution.hooks import HookResult, Middleware

ASK_USER_TOOL_NAME = "ask_user"
ASK_USER_INPUT_PROVIDER_KEY = "ask_user_input_provider"


# Type: (questions: list[dict]) -> (user_answers: dict[str, str] | None, was_cancelled: bool)
UserInputProvider = Callable[[list[dict[str, Any]]], tuple[dict[str, str] | None, bool]]


def _default_cli_provider(questions: list[dict[str, Any]]) -> tuple[dict[str, str] | None, bool]:
    """
    Default CLI provider: print questions to stdout, read answers from stdin.
    For interactive terminal; transport can override with frontend integration.
    """
    user_answers: dict[str, str] = {}
    print("\n--- ask_user: 模型需要你的输入 ---")
    for i, q in enumerate(questions):
        header = q.get("header", f"Q{i + 1}")
        question_text = q.get("question", "")
        qtype = q.get("type", "text")
        placeholder = q.get("placeholder", "")
        print(f"\n  [模型提问] {header}: {question_text}")

        if qtype == "yesno":
            while True:
                ans = input("  [你的回答] (y/n) 请输入 y(是) 或 n(否): ").strip().lower()
                if ans in ("y", "yes"):
                    user_answers[str(i)] = "Yes"
                    break
                if ans in ("n", "no"):
                    user_answers[str(i)] = "No"
                    break
                print("    提示: 请输入 y 或 n")
        elif qtype == "choice":
            options = q.get("options", [])
            print("  可选:")
            for j, opt in enumerate(options):
                print(f"    {j + 1}. {opt.get('label', '')} - {opt.get('description', '')}")
            n = len(options)
            while True:
                try:
                    choice = input(f"  [你的回答] (1-{n}) 请输入序号: ").strip()
                    idx = int(choice)
                    if 1 <= idx <= n:
                        user_answers[str(i)] = options[idx - 1].get("label", "")
                        break
                except ValueError:
                    pass
                print(f"    提示: 请输入 1 到 {n} 之间的数字")
        else:
            hint = f" 示例: {placeholder}" if placeholder else " 请输入文本"
            user_answers[str(i)] = input(f"  [你的回答]{hint}: ").strip()
    print("\n---\n")

    return (user_answers, False)


class AskUserMiddleware(Middleware):
    """
    Middleware that intercepts ask_user tool calls.

    Before the tool executes:
    1. If user_answers not in tool_input (initial call from LLM)
    2. Call user_input_provider(questions) to display and collect user response
    3. Inject user_answers and was_cancelled into tool_input
    4. Tool executes with the injected params
    """

    def __init__(
        self,
        user_input_provider: UserInputProvider | None = None,
        use_global_storage: bool = True,
    ):
        """
        Args:
            user_input_provider: Callable that displays questions and returns (user_answers, was_cancelled).
                When None, uses global_storage provider if set, else default CLI (stdout/stdin).
            use_global_storage: If True, check global_storage for provider at runtime (allows transport override).
        """
        self.user_input_provider = user_input_provider
        self.use_global_storage = use_global_storage

    def _get_provider(self, agent_state: Any) -> UserInputProvider:
        """Resolve the user input provider from config, global_storage, or default CLI."""
        if self.user_input_provider is not None:
            return self.user_input_provider
        if self.use_global_storage and agent_state is not None:
            global_storage = getattr(agent_state, "global_storage", None)
            if global_storage is not None:
                provider = global_storage.get(ASK_USER_INPUT_PROVIDER_KEY)
                if callable(provider):
                    return cast(UserInputProvider, provider)
        return _default_cli_provider

    def before_tool(self, hook_input: Any) -> HookResult:
        """Intercept ask_user: display questions, collect answers, inject into tool_input."""
        if hook_input.tool_name != ASK_USER_TOOL_NAME:
            return HookResult.no_changes()

        tool_input = dict(hook_input.tool_input)
        questions = tool_input.get("questions", [])

        # Skip if already has user_answers (e.g. re-invocation)
        if "user_answers" in tool_input or "was_cancelled" in tool_input:
            return HookResult.no_changes()

        if not questions:
            return HookResult.no_changes()

        provider = self._get_provider(hook_input.agent_state)

        try:
            user_answers, was_cancelled = provider(questions)
            tool_input["user_answers"] = user_answers
            tool_input["was_cancelled"] = was_cancelled
            return HookResult.with_modifications(tool_input=tool_input)
        except Exception:
            # On error (e.g. stdin closed), treat as cancelled
            tool_input["user_answers"] = {}
            tool_input["was_cancelled"] = True
            return HookResult.with_modifications(tool_input=tool_input)
