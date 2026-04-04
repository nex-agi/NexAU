"""Run a 4×4×3×2 thinking-enabled provider switching matrix with Langfuse tracing.

RFC-0014: Live UMP provider-switching validation matrix

This script executes 96 real-provider two-turn cases:
- 4 source providers × 4 target providers × 3 content scenarios × 2 stream modes
- turn 1 on source provider
- turn 2 on target provider (same NexAU session history)
- both turns wrapped in one Langfuse root trace for timeline inspection
- each case tested in both non-streaming and streaming mode

Content scenarios (3rd dimension) — each targets specific UMP block types:

  ┌────────────────────┬───────────────────────────────────────────────┐
  │ Scenario           │ UMP blocks exercised in session history       │
  ├────────────────────┼───────────────────────────────────────────────┤
  │ text_only          │ TextBlock, ReasoningBlock                     │
  │ tool_call          │ ToolUseBlock, ToolResultBlock(str)            │
  │ image_input        │ ImageBlock (user-side multimodal, both turns) │
  └────────────────────┴───────────────────────────────────────────────┘

Combined coverage across all 3 scenarios:
  ✓ TextBlock          — every scenario (user + assistant messages)
  ✓ ImageBlock          — image_input (Turn A: image_1, Turn B: image_2)
  ✓ ReasoningBlock      — every scenario (thinking enabled on all providers)
  ✓ ToolUseBlock        — tool_call
  ✓ ToolResultBlock(str)             — tool_call

Roles exercised: SYSTEM, USER, ASSISTANT, TOOL

Test images:
  - tests/fixtures/images/image_1.jpg  →  used in image_input Turn A
  - tests/fixtures/images/image_2.jpg  →  used in image_input Turn B

Environment variables required:
    OPENAI_CHAT_MODEL / OPENAI_CHAT_BASE_URL / OPENAI_CHAT_API_KEY
    OPENAI_RESPONSES_MODEL / OPENAI_RESPONSES_BASE_URL / OPENAI_RESPONSES_API_KEY
    ANTHROPIC_MODEL / ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY
    GEMINI_MODEL / GEMINI_BASE_URL / GEMINI_API_KEY
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST

Outputs:
    - tests/results/thinking_matrix_langfuse_results.json
    - tests/results/thinking_matrix_langfuse_report.md
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config.config import AgentConfig
from nexau.archs.session.orm.memory_engine import InMemoryDatabaseEngine
from nexau.archs.session.session_manager import SessionManager
from nexau.archs.tool.tool import Tool
from nexau.archs.tracer.adapters.langfuse import LangfuseTracer
from nexau.archs.tracer.context import TraceContext
from nexau.archs.tracer.core import SpanType
from nexau.core.messages import ImageBlock, Message, Role, TextBlock

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_TESTS_DIR = _SCRIPT_DIR.parent

RESULTS_JSON = "tests/results/thinking_matrix_langfuse_results.json"
RESULTS_REPORT = "tests/results/thinking_matrix_langfuse_report.md"
CHECKPOINT_FILE = "tests/results/thinking_matrix_checkpoint.json"

# Image 1: cos330° + tan600° 选择题 — used in image_input Turn A
_IMG_1 = _TESTS_DIR / "fixtures" / "images" / "image_1.jpg"
# Image 2: y=aˣ 在 [0,1] 上最大值+最小值=3 求 a 选择题 — used in image_input Turn B
_IMG_2 = _TESTS_DIR / "fixtures" / "images" / "image_2.jpg"


def _load_image_base64(path: Path) -> str:
    """Load an image file and return its base64-encoded content.

    RFC-0014: 运行时加载测试图片，转为 base64 嵌入 UMP ImageBlock
    """
    if not path.exists():
        raise FileNotFoundError(f"Test image not found: {path}")
    return base64.b64encode(path.read_bytes()).decode()


# ---------------------------------------------------------------------------
# Scenario 1: text_only — 幂级数求和 + 模运算
# UMP: TextBlock + ReasoningBlock
#
# Turn A: Σ(n=1,∞) n³/2ⁿ  需要对几何级数连续三次执行 x·d/dx 算子
#   Σ xⁿ = 1/(1-x)
#   x·d/dx → Σ n·xⁿ = x/(1-x)²
#   x·d/dx → Σ n²·xⁿ = x(1+x)/(1-x)³
#   x·d/dx → Σ n³·xⁿ = x(1+4x+x²)/(1-x)⁴
#   代入 x=½ → (1/2)(13/4)/(1/16) = 26
#
# Turn B: (7²⁶ + 3²⁶) mod 100
#   ord(7, 100) = 4 → 7²⁶ = 7^(4·6+2) ≡ 7² = 49
#   ord(3, 100) = 20 → 3²⁶ = 3^(20+6) ≡ 3⁶ = 729 ≡ 29
#   49 + 29 = 78
# ---------------------------------------------------------------------------

_TEXT_ONLY_TASK_A = (
    "Compute the exact value of the infinite series Σ(n=1 to ∞) n³ / 2ⁿ. "
    "Start from the geometric series Σ(n=0,∞) xⁿ = 1/(1−x) for |x|<1 "
    "and apply the operator x·(d/dx) three times to build closed-form "
    "expressions for Σ n·xⁿ, Σ n²·xⁿ, and Σ n³·xⁿ. "
    "Then evaluate at x = 1/2. "
    "End with Final A: <number>."
)
# x(1+4x+x²)/(1-x)⁴ at x=½ → 26
_TEXT_ONLY_EXPECTED_A = "26"

_TEXT_ONLY_TASK_B = (
    "Let S denote the previous series sum. "
    "Compute the remainder when 7^S + 3^S is divided by 100. "
    "Find the multiplicative orders of 7 and 3 modulo 100 "
    "(use Euler's theorem as a starting point), reduce the exponents, "
    "then add the residues. "
    "End with Final B: <number>."
)
# 7^26 mod 100 = 49, 3^26 mod 100 = 29 → 49+29 = 78
_TEXT_ONLY_EXPECTED_B = "78"


# ---------------------------------------------------------------------------
# Scenario 2: tool_call — 幂运算 + 取模
# UMP: ToolUseBlock + ToolResultBlock(str)
# ---------------------------------------------------------------------------

_TOOL_CALL_TASK_A = (
    "Use the calculator tool to evaluate this expression step by step: "
    "(2**10 + 3**7) * 17 - 4999. "
    "Report the calculator result and end with Final A: <number>."
)
# 2^10=1024, 3^7=2187, (1024+2187)*17-4999 = 3211*17-4999 = 54587-4999 = 49588
_TOOL_CALL_EXPECTED_A = "49588"

_TOOL_CALL_TASK_B = (
    "Take the result from the previous task and use the calculator to find its remainder when divided by 256. End with Final B: <number>."
)
# 49588 mod 256 = 49588 - 193*256 = 49588 - 49408 = 180
_TOOL_CALL_EXPECTED_B = "180"


# ---------------------------------------------------------------------------
# Scenario 4: image_input — 用户消息含图片（两轮均含图片）
# UMP: ImageBlock (user-side multimodal)
#
# Turn A: Provider A sends Image 1 (cos330° + tan600° 选择题)
#   cos330° = √3/2,  tan600° = tan240° = √3
#   cos330° + tan600° = 3√3/2 ≈ 2.598 → 2.6
#
# Turn B: Provider B sends Image 2 (y=aˣ 在 [0,1] 上最大值+最小值=3 选择题)
#   若 a>1: max=a, min=1 → a+1=3 → a=2
#   若 0<a<1: max=1, min=a → 1+a=3 → a=2 (矛盾)
#   答案: a=2
# ---------------------------------------------------------------------------

_IMAGE_INPUT_TASK_A = (
    "Look at the image. It shows a math multiple-choice problem. "
    "Solve it step by step. "
    "Give the decimal value rounded to 1 decimal place. "
    "End with Final A: <number>."
)
# 3√3/2 ≈ 2.598 → rounded to 1dp → 2.6
_IMAGE_INPUT_EXPECTED_A = "2.6"

_IMAGE_INPUT_TASK_B = (
    "Look at this new image. It shows a different math problem. Solve it step by step and find the answer. End with Final B: <number>."
)
# a > 1 → max=a, min=1 → a+1=3 → a=2
_IMAGE_INPUT_EXPECTED_B = "2"


# ---------------------------------------------------------------------------
# Content Scenario definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContentScenario:
    """Describes the 3rd dimension of the matrix.

    RFC-0014: 内容场景定义

    每个场景针对不同的 UMP block 类型组合，确保所有 block 类型
    在跨 provider 切换时的序列化和反序列化都被测试到。
    """

    name: str
    description: str
    task_a: str
    task_b: str
    expected_a: str
    expected_b: str
    needs_calculator: bool
    needs_image_input: bool
    ump_blocks_exercised: list[str]


SCENARIOS: list[ContentScenario] = [
    ContentScenario(
        name="text_only",
        description="Power series Σn³/2ⁿ + modular arithmetic — TextBlock + ReasoningBlock",
        task_a=_TEXT_ONLY_TASK_A,
        task_b=_TEXT_ONLY_TASK_B,
        expected_a=_TEXT_ONLY_EXPECTED_A,
        expected_b=_TEXT_ONLY_EXPECTED_B,
        needs_calculator=False,
        needs_image_input=False,
        ump_blocks_exercised=["TextBlock", "ReasoningBlock"],
    ),
    ContentScenario(
        name="tool_call",
        description="Power + modulo via calculator — ToolUseBlock + ToolResultBlock(str)",
        task_a=_TOOL_CALL_TASK_A,
        task_b=_TOOL_CALL_TASK_B,
        expected_a=_TOOL_CALL_EXPECTED_A,
        expected_b=_TOOL_CALL_EXPECTED_B,
        needs_calculator=True,
        needs_image_input=False,
        ump_blocks_exercised=[
            "TextBlock",
            "ReasoningBlock",
            "ToolUseBlock",
            "ToolResultBlock(str)",
        ],
    ),
    ContentScenario(
        name="image_input",
        description="User sends trig problem image — ImageBlock in user message",
        task_a=_IMAGE_INPUT_TASK_A,
        task_b=_IMAGE_INPUT_TASK_B,
        expected_a=_IMAGE_INPUT_EXPECTED_A,
        expected_b=_IMAGE_INPUT_EXPECTED_B,
        needs_calculator=False,
        needs_image_input=True,
        ump_blocks_exercised=[
            "TextBlock",
            "ReasoningBlock",
            "ImageBlock(user_message)",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    name: str
    api_type: str
    model: str
    base_url: str
    api_key: str


@dataclass
class MatrixCaseResult:
    case_name: str
    source_provider: str
    target_provider: str
    scenario: str
    stream_mode: bool
    session_id: str
    source_api_type: str
    target_api_type: str
    first_turn_ok: bool
    second_turn_ok: bool
    first_turn_excerpt: str
    second_turn_excerpt: str
    final_a_detected: bool
    final_b_detected: bool
    status: str
    error: str | None
    duration_seconds: float


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    RFC-0014: 计算器工具，用于 tool_call 场景测试

    安全限制：只允许基本数学运算字符；自动将 ^ 转为 ** 以兼容不同写法。
    """
    # 1. 兼容 ^ 表示幂运算
    expression = expression.replace("^", "**")

    # 2. 安全检查：只允许数字和基本运算符
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return f"Error: expression contains disallowed characters: {expression}"
    try:
        result = eval(expression)  # noqa: S307 — controlled test input
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


def _build_calculator_tool() -> Tool:
    """Build the calculator Tool instance."""
    return Tool(
        name="calculator",
        description=(
            "Evaluate a mathematical expression and return the numeric result. "
            "Use Python syntax: ** for power (e.g. 2**10), % for modulo. "
            "Input: {expression: string}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": ("Mathematical expression to evaluate, e.g. '(2**10 + 3**7) * 17 - 4999' or '49588 % 256'"),
                },
            },
            "required": ["expression"],
        },
        implementation=_calculator,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _provider_configs() -> list[ProviderConfig]:
    """Load provider configs from env vars.

    RFC-0014: 软加载 — 缺少环境变量的 provider 被跳过而非报错

    只返回所有环境变量都已配置的 provider，方便只测试部分 provider。
    """
    all_specs: list[tuple[str, str, str, str, str]] = [
        # (name, api_type, MODEL_env, BASE_URL_env, API_KEY_env)
        ("openai-chat", "openai_chat_completion", "OPENAI_CHAT_MODEL", "OPENAI_CHAT_BASE_URL", "OPENAI_CHAT_API_KEY"),
        ("openai-responses", "openai_responses", "OPENAI_RESPONSES_MODEL", "OPENAI_RESPONSES_BASE_URL", "OPENAI_RESPONSES_API_KEY"),
        ("anthropic", "anthropic_chat_completion", "ANTHROPIC_MODEL", "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"),
        ("gemini", "gemini_rest", "GEMINI_MODEL", "GEMINI_BASE_URL", "GEMINI_API_KEY"),
    ]
    providers: list[ProviderConfig] = []
    for name, api_type, model_env, url_env, key_env in all_specs:
        model = os.getenv(model_env, "").strip()
        base_url = os.getenv(url_env, "").strip()
        api_key = os.getenv(key_env, "").strip()
        if model and base_url and api_key:
            providers.append(ProviderConfig(name=name, api_type=api_type, model=model, base_url=base_url, api_key=api_key))
        else:
            missing = [v for v, val in [(model_env, model), (url_env, base_url), (key_env, api_key)] if not val]
            print(f"  ⏭️  Provider '{name}' skipped — missing env: {', '.join(missing)}")
    if not providers:
        raise RuntimeError("No providers configured. Set at least one provider's env vars.")
    return providers


def _langfuse_config() -> tuple[str, str, str]:
    return (
        _required_env("LANGFUSE_PUBLIC_KEY"),
        _required_env("LANGFUSE_SECRET_KEY"),
        _required_env("LANGFUSE_HOST"),
    )


def _build_llm_config(provider: ProviderConfig, *, stream: bool = False) -> LLMConfig:
    """Build an LLMConfig with thinking enabled and generous max_tokens.

    RFC-0014: 构建带 thinking 参数的 LLM 配置

    max_tokens 设为 16384，确保 thinking 推理和最终回答都有足够空间，
    避免截断导致 Final A / Final B 标记丢失。
    stream 参数控制是否使用流式模式。
    """
    kwargs: dict = {
        "model": provider.model,
        "base_url": provider.base_url,
        "api_key": provider.api_key,
        "api_type": provider.api_type,
        "temperature": 0.0,
        "max_tokens": 16384,
        "max_retries": 1,
        "stream": stream,
        # HTTP 级别超时 — 防止僵尸线程的 HTTP 连接堆积打爆 proxy
        # streaming 模式下超时放宽到 300s，image + thinking 场景首 token 延迟较大
        "timeout": 300.0 if stream else 180.0,
    }

    # 1. Provider-specific thinking parameters
    if provider.api_type == "openai_chat_completion":
        kwargs["reasoning_effort"] = "high"
    elif provider.api_type == "openai_responses":
        kwargs["reasoning"] = {"effort": "high"}
        kwargs["include"] = ["reasoning.encrypted_content"]
    elif provider.api_type == "anthropic_chat_completion":
        # Anthropic 要求 thinking 开启时 temperature 必须为 1
        kwargs["temperature"] = 1.0
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8192}
    elif provider.api_type == "gemini_rest":
        kwargs["thinkingConfig"] = {"includeThoughts": True, "thinkingBudget": 8192}

    return LLMConfig(**kwargs)


def _build_tools_for_scenario(scenario: ContentScenario) -> list[Tool]:
    """Build the tool list required by a given content scenario.

    RFC-0014: 根据场景构建工具列表
    """
    tools: list[Tool] = []
    if scenario.needs_calculator:
        tools.append(_build_calculator_tool())
    return tools


def _build_user_message_a(scenario: ContentScenario) -> str | list[Message]:
    """Build the turn-1 user message for a given scenario.

    RFC-0014: 构建 Turn A 用户消息

    对于 image_input 场景返回 list[Message]，内含一条 User Message
    同时包含 TextBlock 和 ImageBlock（base64 编码的 tests/image_1.jpg）；
    其余场景返回纯文本 str。
    """
    if scenario.needs_image_input:
        img_b64 = _load_image_base64(_IMG_1)
        return [
            Message(
                role=Role.USER,
                content=[
                    TextBlock(text=scenario.task_a),
                    ImageBlock(base64=img_b64, mime_type="image/jpeg"),
                ],
            ),
        ]
    return scenario.task_a


def _build_user_message_b(scenario: ContentScenario) -> str | list[Message]:
    """Build the turn-2 user message for a given scenario.

    RFC-0014: 构建 Turn B 用户消息

    对于 image_input 场景返回 list[Message]，内含一条 User Message
    同时包含 TextBlock 和 ImageBlock（base64 编码的 tests/image_2.jpg）；
    其余场景返回纯文本 str。

    这确保两轮都带 ImageBlock，真正测试跨 provider 切换时
    用户侧 ImageBlock 的序列化和反序列化。
    """
    if scenario.needs_image_input:
        img_b64 = _load_image_base64(_IMG_2)
        return [
            Message(
                role=Role.USER,
                content=[
                    TextBlock(text=scenario.task_b),
                    ImageBlock(base64=img_b64, mime_type="image/jpeg"),
                ],
            ),
        ]
    return scenario.task_b


def _make_agent(
    provider: ProviderConfig,
    scenario: ContentScenario,
    *,
    stream: bool,
    session_manager: SessionManager,
    session_id: str,
    tracer: LangfuseTracer,
) -> Agent:
    """Create an Agent configured for a specific provider + scenario combination.

    RFC-0014: 根据 provider、scenario 和 stream mode 创建 Agent
    """
    system_prompt = (
        "You are a precise computation assistant. "
        "Think step by step before answering. "
        "Keep calculations concise and preserve prior context "
        "when continuing a conversation. "
        "If you have tools available, use them when appropriate."
    )

    tools = _build_tools_for_scenario(scenario)

    return Agent(
        config=AgentConfig(
            name=f"thinking-matrix-{provider.name}-{scenario.name}",
            system_prompt=system_prompt,
            llm_config=_build_llm_config(provider, stream=stream),
            tracers=[tracer],
            tools=tools,
        ),
        session_manager=session_manager,
        user_id="thinking-matrix-user",
        session_id=session_id,
    )


def _excerpt(text: object) -> str:
    """Create a short readable excerpt from agent output."""
    if not isinstance(text, str):
        text = str(text) if text else ""
    rendered = text.replace("\n", " ")
    return rendered[:240]


# ---------------------------------------------------------------------------
# Preflight health check
# ---------------------------------------------------------------------------


@dataclass
class HealthCheckResult:
    """Result of a single provider or service health check."""

    name: str
    alive: bool
    latency_ms: float
    detail: str


def _check_provider(provider: ProviderConfig) -> HealthCheckResult:
    """Send a minimal request to a provider to verify API key and connectivity.

    RFC-0014: 验活检查 — 通过真实 Agent 管线发送 "say hi"

    走完整 Agent.run() 路径（含 LLMCaller、serializer），不做任何 mock。
    max_tokens 设为 16 以节省 token，stream=False 以快速返回。
    """
    start = time.time()
    try:
        llm_kwargs: dict = {
            "model": provider.model,
            "base_url": provider.base_url,
            "api_key": provider.api_key,
            "api_type": provider.api_type,
            "temperature": 0.0,
            "max_tokens": 16,
            "max_retries": 1,
            "stream": False,
        }
        # Anthropic 要求 thinking 关闭时才能 temperature=0，这里不开 thinking
        if provider.api_type == "anthropic_chat_completion":
            llm_kwargs["temperature"] = 1.0

        agent = Agent(
            config=AgentConfig(
                name=f"health-check-{provider.name}",
                system_prompt="Reply with OK.",
                llm_config=LLMConfig(**llm_kwargs),
            ),
            session_manager=SessionManager(engine=InMemoryDatabaseEngine()),
            user_id="health-check",
            session_id=f"health-{uuid.uuid4().hex[:6]}",
        )
        resp = agent.run(message="Say OK.")
        latency = (time.time() - start) * 1000
        text = str(resp) if resp else ""
        if text:
            return HealthCheckResult(
                name=provider.name,
                alive=True,
                latency_ms=round(latency, 1),
                detail=text[:60].replace("\n", " "),
            )
        return HealthCheckResult(
            name=provider.name,
            alive=False,
            latency_ms=round(latency, 1),
            detail="Empty response",
        )
    except Exception as exc:
        latency = (time.time() - start) * 1000
        return HealthCheckResult(
            name=provider.name,
            alive=False,
            latency_ms=round(latency, 1),
            detail=f"{type(exc).__name__}: {str(exc)[:80]}",
        )


def _check_langfuse(public_key: str, secret_key: str, host: str) -> HealthCheckResult:
    """Verify Langfuse API connectivity by listing traces.

    RFC-0014: Langfuse 验活 — 调用 /api/public/traces 确认凭证有效
    """
    start = time.time()
    try:
        resp = requests.get(
            f"{host}/api/public/traces",
            params={"limit": 1},
            auth=(public_key, secret_key),
            timeout=10,
        )
        latency = (time.time() - start) * 1000
        if resp.status_code == 200:
            return HealthCheckResult(
                name="langfuse",
                alive=True,
                latency_ms=round(latency, 1),
                detail=f"HTTP 200, host={host}",
            )
        return HealthCheckResult(
            name="langfuse",
            alive=False,
            latency_ms=round(latency, 1),
            detail=f"HTTP {resp.status_code}: {resp.text[:80]}",
        )
    except Exception as exc:
        latency = (time.time() - start) * 1000
        return HealthCheckResult(
            name="langfuse",
            alive=False,
            latency_ms=round(latency, 1),
            detail=f"{type(exc).__name__}: {str(exc)[:80]}",
        )


def _run_preflight(
    providers: list[ProviderConfig],
    public_key: str,
    secret_key: str,
    host: str,
) -> tuple[list[HealthCheckResult], bool]:
    """Run preflight health checks on all providers + Langfuse.

    RFC-0014: 预检验活 — 在跑 96-case 矩阵前确认所有服务可达

    Returns:
        (results, all_ok) — all_ok 为 False 时不会中止，仅打印警告。
    """
    print("\n" + "=" * 70)
    print("  PREFLIGHT HEALTH CHECK")
    print("=" * 70)

    results: list[HealthCheckResult] = []

    # 1. 检查 Langfuse
    lf_result = _check_langfuse(public_key, secret_key, host)
    results.append(lf_result)
    status_icon = "✅" if lf_result.alive else "❌"
    print(f"  {status_icon} langfuse    ({lf_result.latency_ms:.0f}ms) {lf_result.detail}")

    # 2. 检查每个 provider
    for provider in providers:
        result = _check_provider(provider)
        results.append(result)
        status_icon = "✅" if result.alive else "❌"
        print(f"  {status_icon} {result.name:20s} ({result.latency_ms:.0f}ms) {result.detail}")

    all_ok = all(r.alive for r in results)
    alive_count = sum(1 for r in results if r.alive)
    total_count = len(results)

    print(f"\n  Result: {alive_count}/{total_count} services alive", end="")
    if all_ok:
        print(" — all clear ✅")
    else:
        dead = [r.name for r in results if not r.alive]
        print(f" — DEGRADED ⚠️  (dead: {', '.join(dead)})")
        print("  Continuing anyway — dead providers will be classified as 'blocked'.")
    print("=" * 70)

    return results, all_ok


# ---------------------------------------------------------------------------
# Checkpoint helpers — 断点重传
# ---------------------------------------------------------------------------


def _case_key(source: str, target: str, scenario: str, stream: bool = False) -> str:
    """Deterministic key for a case, used as checkpoint dict key."""
    stream_tag = "stream" if stream else "nostream"
    return f"{source}__{target}__{scenario}__{stream_tag}"


def _save_checkpoint(
    results: list[MatrixCaseResult],
    checkpoint_path: str = CHECKPOINT_FILE,
) -> None:
    """Incrementally save completed results to checkpoint file.

    RFC-0014: 每完成一个 case 就写入 checkpoint，支持中断后断点恢复

    合并策略：先读取已有 checkpoint，再用本次 results 覆盖对应 key，
    确保用 --source/--target/--scenario 过滤跑子集时不丢失已有结果。
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 读取已有 checkpoint（如果存在）
    existing_cases: dict[str, dict[str, object]] = {}
    if path.exists():
        try:
            with open(path) as f:
                old = json.load(f)
            if old.get("_checkpoint"):
                existing_cases = old.get("cases", {})
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    # 2. 用本次 results 覆盖对应 key（新结果优先）
    for r in results:
        existing_cases[_case_key(r.source_provider, r.target_provider, r.scenario, r.stream_mode)] = asdict(r)

    data: dict[str, object] = {
        "_checkpoint": True,
        "_saved_at": datetime.now(UTC).isoformat(),
        "_count": len(existing_cases),
        "cases": existing_cases,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_checkpoint(
    checkpoint_path: str = CHECKPOINT_FILE,
) -> dict[str, MatrixCaseResult]:
    """Load completed results from checkpoint file.

    RFC-0014: 从 checkpoint 文件恢复已完成的 case，跳过重复执行

    Returns:
        dict mapping case_key → MatrixCaseResult
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        if not data.get("_checkpoint"):
            return {}
        cases_raw: dict[str, dict[str, object]] = data.get("cases", {})
        restored: dict[str, MatrixCaseResult] = {}
        for key, val in cases_raw.items():
            # 只恢复 passed 的 case，其他状态重跑
            if val.get("status") == "passed":
                # 兼容旧 checkpoint：缺少 stream_mode 字段时默认为 False
                if "stream_mode" not in val:
                    val["stream_mode"] = False
                restored[key] = MatrixCaseResult(**val)  # type: ignore[arg-type]
        return restored
    except (json.JSONDecodeError, TypeError, KeyError):
        return {}


def _classify_status(
    first_ok: bool,
    second_ok: bool,
    error: str | None,
) -> str:
    """Classify case outcome: passed / blocked / failed."""
    # 1. Happy path
    if first_ok and second_ok and not error:
        return "passed"

    # 2. Infrastructure block detection
    if error:
        lowered = error.lower()
        if any(
            kw in lowered
            for kw in (
                "error code: 429",
                "负载已饱和",
                "error code: 401",
                "无效的令牌",
                "invalid api key",
                "invalid_api_key",
                "error code: 403",
                "temporarily blocked",
                "content policy",
                "error code: 500",
                "error code: 503",
                "model_not_found",
                "无可用渠道",
                "internal_server_error",
                "upstream error",
            )
        ):
            return "blocked"

    # 3. Everything else is a real failure
    return "failed"


def _run_case(
    source: ProviderConfig,
    target: ProviderConfig,
    scenario: ContentScenario,
    public_key: str,
    secret_key: str,
    host: str,
    *,
    stream: bool = False,
) -> MatrixCaseResult:
    """Run a single two-turn case and collect results.

    RFC-0014: 执行单个两轮测试用例

    对于不同 content scenario，session history 中产生的 UMP block 组合不同：
    - text_only:          user[TextBlock] → assistant[ReasoningBlock, TextBlock]
    - tool_call:          + assistant[ToolUseBlock] → tool[ToolResultBlock(str)]
    - image_input:        user[TextBlock, ImageBlock] → assistant (both turns have images)

    Turn 2 在不同 provider 上序列化上述 history，验证跨 provider 兼容性。
    stream=True 时测试流式事件拼接逻辑，验证每种 API 的 streaming aggregator。
    """
    stream_tag = "stream" if stream else "nostream"
    case_name = f"thinking-matrix:{source.name}-to-{target.name}:{scenario.name}:{stream_tag}"
    session_id = f"tm-{source.name}-{target.name}-{scenario.name}-{stream_tag}-{uuid.uuid4().hex[:6]}"

    # 1. 创建 Langfuse tracer
    tracer = LangfuseTracer(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        session_id=session_id,
        tags=[
            "thinking-matrix-96",
            source.name,
            target.name,
            source.api_type,
            target.api_type,
            scenario.name,
            stream_tag,
        ],
        metadata={
            "test": "thinking_matrix_langfuse_96",
            "source_provider": source.name,
            "target_provider": target.name,
            "source_api_type": source.api_type,
            "target_api_type": target.api_type,
            "scenario": scenario.name,
            "stream_mode": stream,
            "ump_blocks": scenario.ump_blocks_exercised,
        },
        debug=False,
    )

    # 2. 创建 session manager
    session_manager = SessionManager(engine=InMemoryDatabaseEngine())

    first_excerpt = ""
    second_excerpt = ""
    first_ok = False
    second_ok = False
    error: str | None = None
    start_time = time.time()

    try:
        with TraceContext(
            tracer,
            case_name,
            SpanType.AGENT,
            inputs={
                "task_a": scenario.task_a,
                "task_b": scenario.task_b,
                "scenario": scenario.name,
                "ump_blocks": scenario.ump_blocks_exercised,
            },
        ):
            # 3. Turn 1: source provider
            source_agent = _make_agent(
                source,
                scenario,
                stream=stream,
                session_manager=session_manager,
                session_id=session_id,
                tracer=tracer,
            )
            user_msg_a = _build_user_message_a(scenario)
            try:
                first = source_agent.run(message=user_msg_a)
                first_excerpt = _excerpt(first)
                first_ok = scenario.expected_a in str(first)
                if not first_ok:
                    error = f"[Turn1 wrong answer] expected '{scenario.expected_a}' not found in output: {first_excerpt[:120]}"
            except Exception as exc:
                error = f"[Turn1 exception] {type(exc).__name__}: {exc}"

            # 4. Turn 2: target provider (same session → history preserved)
            if not error or first_ok:
                target_agent = _make_agent(
                    target,
                    scenario,
                    stream=stream,
                    session_manager=session_manager,
                    session_id=session_id,
                    tracer=tracer,
                )
                try:
                    second = target_agent.run(message=_build_user_message_b(scenario))
                    second_excerpt = _excerpt(second)
                    second_ok = scenario.expected_b in str(second)
                    if not second_ok:
                        t2_err = f"[Turn2 wrong answer] expected '{scenario.expected_b}' not found in output: {second_excerpt[:120]}"
                        error = f"{error} | {t2_err}" if error else t2_err
                except Exception as exc:
                    t2_err = f"[Turn2 exception] {type(exc).__name__}: {exc}"
                    error = f"{error} | {t2_err}" if error else t2_err
    except Exception as exc:
        error = f"[TraceContext exception] {type(exc).__name__}: {exc}"

    # 5. Flush + shutdown traces (with timeout to prevent hanging)
    #
    # Langfuse SDK 的 flush() 内部调用 Queue.join()（无 timeout），
    # 如果 media upload queue 有未完成任务（如图片上传超时），会永远阻塞。
    # 因此用 daemon thread + timeout 保护。
    #
    # 关键：无论 flush 是否超时，都必须调用 shutdown() 来终止后台线程，
    # 否则上一个 tracer 的 zombie 线程会与下一个 tracer 产生死锁。
    import threading

    def _flush_only() -> None:
        try:
            tracer.flush()
        except Exception:
            pass

    t = threading.Thread(target=_flush_only, daemon=True)
    t.start()
    t.join(timeout=30)
    if t.is_alive():
        print("    ⚠️  Langfuse flush timed out (30s), forcing shutdown...")

    # 始终执行 shutdown，即使 flush 超时也要清理后台线程
    def _shutdown_only() -> None:
        try:
            tracer.shutdown()
        except Exception:
            pass

    t2 = threading.Thread(target=_shutdown_only, daemon=True)
    t2.start()
    t2.join(timeout=10)
    if t2.is_alive():
        print("    ⚠️  Langfuse shutdown timed out (10s), continuing...")

    duration = time.time() - start_time

    status = _classify_status(
        first_ok=first_ok,
        second_ok=second_ok,
        error=error,
    )

    return MatrixCaseResult(
        case_name=case_name,
        source_provider=source.name,
        target_provider=target.name,
        scenario=scenario.name,
        stream_mode=stream,
        session_id=session_id,
        source_api_type=source.api_type,
        target_api_type=target.api_type,
        first_turn_ok=first_ok,
        second_turn_ok=second_ok,
        first_turn_excerpt=first_excerpt,
        second_turn_excerpt=second_excerpt,
        final_a_detected=first_ok,
        final_b_detected=second_ok,
        status=status,
        error=error,
        duration_seconds=round(duration, 2),
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _render_ump_coverage_table() -> list[str]:
    """Render the UMP block-type coverage matrix.

    RFC-0014: 生成 UMP 覆盖矩阵，证明所有 block 类型被完整覆盖
    """
    return [
        "## UMP Block-Type Coverage Matrix",
        "",
        "All 5 block types in `BlockType = TextBlock | ImageBlock | ReasoningBlock | ToolUseBlock | ToolResultBlock` are covered.",
        "",
        "| UMP Block Type | Variant | Covered By Scenario | Position in History |",
        "|---|---|---|---|",
        "| **TextBlock** | — | all scenarios | user msg, assistant msg |",
        "| **ImageBlock** | user-side (base64) | `image_input` | user msg Turn A (image_1.jpg) + Turn B (image_2.jpg) |",
        "| **ReasoningBlock** | — | all scenarios | assistant msg (thinking enabled) |",
        "| **ToolUseBlock** | — | `tool_call` | assistant msg |",
        "| **ToolResultBlock** | `content: str` | `tool_call` | tool-role msg (calculator result) |",
        "",
        "### Roles Exercised",
        "",
        "| Role | Covered By |",
        "|---|---|",
        "| `SYSTEM` | All scenarios (system prompt) |",
        "| `USER` | All scenarios (user message) |",
        "| `ASSISTANT` | All scenarios (model response) |",
        "| `TOOL` | `tool_call` |",
        "",
    ]


def _render_report(results: list[MatrixCaseResult]) -> str:
    """Render a Markdown report from results.

    RFC-0014: 生成 96-case 测试报告（含 streaming 维度）
    """
    passed = sum(1 for r in results if r.status == "passed")
    blocked = sum(1 for r in results if r.status == "blocked")
    failed = sum(1 for r in results if r.status == "failed")

    lines: list[str] = [
        "# 96-Case Thinking Provider × Content Scenario × Stream Mode Matrix Report",
        "",
        f"- **Scope**: 4 source × 4 target × 3 scenarios × 2 stream modes = {len(results)} cases with thinking + Langfuse tracing",
        f"- **Generated At**: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- **Total Cases**: {len(results)}",
        f"- **Passed**: {passed}",
        f"- **Blocked**: {blocked}",
        f"- **Failed**: {failed}",
        "",
    ]

    # UMP coverage matrix
    lines.extend(_render_ump_coverage_table())

    # Scenario descriptions
    lines.extend(
        [
            "## Scenario Definitions",
            "",
            "| Scenario | Task A | Expected A | Task B (follow-up) | Expected B |",
            "|----------|--------|------------|-------------------|------------|",
        ]
    )
    for sc in SCENARIOS:
        task_a_short = sc.task_a[:60].replace("|", "\\|") + "…"
        task_b_short = sc.task_b[:60].replace("|", "\\|") + "…"
        lines.append(f"| `{sc.name}` | {task_a_short} | `{sc.expected_a}` | {task_b_short} | `{sc.expected_b}` |")
    lines.append("")

    # Summary table
    lines.extend(
        [
            "## Summary Table",
            "",
            "| # | Source → Target | Scenario | Stream | Status | Duration | Note |",
            "|---|----------------|----------|--------|--------|----------|------|",
        ]
    )

    for i, result in enumerate(results, 1):
        if result.error:
            note = result.error.replace("\n", " ")[:100]
        elif result.second_turn_excerpt:
            note = result.second_turn_excerpt[:100]
        else:
            note = result.first_turn_excerpt[:100]

        stream_label = "stream" if result.stream_mode else "non-stream"
        lines.append(
            f"| {i} | {result.source_provider} → {result.target_provider} "
            f"| `{result.scenario}` | {stream_label} "
            f"| **{result.status}** | {result.duration_seconds:.1f}s | {note} |"
        )

    # Per-scenario summary
    lines.extend(
        [
            "",
            "## Per-Scenario Summary",
            "",
        ]
    )

    for scenario in SCENARIOS:
        sc_results = [r for r in results if r.scenario == scenario.name]
        sc_passed = sum(1 for r in sc_results if r.status == "passed")
        sc_blocked = sum(1 for r in sc_results if r.status == "blocked")
        sc_failed = sum(1 for r in sc_results if r.status == "failed")
        blocks = ", ".join(scenario.ump_blocks_exercised)
        lines.append(f"### `{scenario.name}` — {sc_passed} passed / {sc_blocked} blocked / {sc_failed} failed")
        lines.append(f"UMP blocks: {blocks}")
        lines.append("")

    # Per-stream-mode summary
    lines.extend(
        [
            "## Per-Stream-Mode Summary",
            "",
        ]
    )
    for stream_val, label in [(False, "non-stream"), (True, "stream")]:
        sm_results = [r for r in results if r.stream_mode == stream_val]
        sm_passed = sum(1 for r in sm_results if r.status == "passed")
        sm_blocked = sum(1 for r in sm_results if r.status == "blocked")
        sm_failed = sum(1 for r in sm_results if r.status == "failed")
        lines.append(f"### `{label}` — {sm_passed} passed / {sm_blocked} blocked / {sm_failed} failed")
        lines.append("")

    # Detail per case
    lines.extend(
        [
            "## Detail per Case",
            "",
        ]
    )

    for result in results:
        sc = next(s for s in SCENARIOS if s.name == result.scenario)
        stream_label = "stream" if result.stream_mode else "non-stream"
        lines.extend(
            [
                f"### {result.source_provider} → {result.target_provider} [`{result.scenario}`] [{stream_label}] — **{result.status}**",
                "",
                f"- **Session ID**: `{result.session_id}`",
                f"- **API types**: {result.source_api_type} → {result.target_api_type}",
                f"- **Scenario**: `{result.scenario}`",
                f"- **Stream mode**: {stream_label}",
                f"- **UMP blocks**: {', '.join(sc.ump_blocks_exercised)}",
                f"- **Turn 1 OK**: {result.first_turn_ok}  |  **Turn 2 OK**: {result.second_turn_ok}",
                f"- **Duration**: {result.duration_seconds:.2f}s",
            ]
        )
        if result.error:
            lines.append(f"- **Error**: `{result.error}`")
        lines.extend(
            [
                f"- **Turn 1 excerpt**: {result.first_turn_excerpt}",
                f"- **Turn 2 excerpt**: {result.second_turn_excerpt}",
                "",
            ]
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for filtering the test matrix.

    RFC-0014: CLI 过滤参数 — 支持自定义测试范围

    Examples::

        # 跑全量 96 case (4×4×3×2)
        uv run python tests/scripts/run_thinking_matrix_langfuse.py

        # 只跑 text_only + tool_call 场景，只用 openai-chat 和 openai-responses
        uv run python tests/scripts/run_thinking_matrix_langfuse.py \\
            --scenario text_only tool_call \\
            --source openai-chat openai-responses \\
            --target openai-chat openai-responses

        # 只跑 streaming 模式
        uv run python tests/scripts/run_thinking_matrix_langfuse.py --stream stream

        # 只跑 non-streaming 模式
        uv run python tests/scripts/run_thinking_matrix_langfuse.py --stream nostream

        # 单场景快速验证
        uv run python tests/scripts/run_thinking_matrix_langfuse.py \\
            --scenario text_only --source openai-chat --target openai-chat

        # 跳过预检验活
        uv run python tests/scripts/run_thinking_matrix_langfuse.py --skip-preflight
    """
    p = argparse.ArgumentParser(
        description="Run the 4×4×3×2 thinking matrix with Langfuse tracing (filterable).",
    )
    p.add_argument(
        "--scenario",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Only run these scenarios (e.g. text_only tool_call). Default: all 3.",
    )
    p.add_argument(
        "--source",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Only use these providers as source (e.g. openai-chat openai-responses). Default: all configured.",
    )
    p.add_argument(
        "--target",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Only use these providers as target. Default: all configured.",
    )
    p.add_argument(
        "--stream",
        nargs="+",
        default=None,
        metavar="MODE",
        help="Only run these stream modes: 'stream' and/or 'nostream'. Default: both.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint file, skipping already-completed cases.",
    )
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the preflight health check.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # 0. 预检测试图片是否存在
    for img_path, label in [
        (_IMG_1, "image_1 (cos330°+tan600°)"),
        (_IMG_2, "image_2 (y=aˣ)"),
    ]:
        if not img_path.exists():
            print(f"ERROR: Test image for '{label}' not found: {img_path}")
            return 1
        print(f"  ✓ {label} image: {img_path.name} ({img_path.stat().st_size // 1024}KB)")

    public_key, secret_key, host = _langfuse_config()
    providers = _provider_configs()

    # 1. 按 CLI 参数过滤 provider 和 scenario
    if args.source:
        providers_src = [p for p in providers if p.name in args.source]
        skipped = set(args.source) - {p.name for p in providers_src}
        if skipped:
            print(f"  ⚠️  --source filter: names not found in configured providers: {skipped}")
    else:
        providers_src = providers

    if args.target:
        providers_tgt = [p for p in providers if p.name in args.target]
        skipped = set(args.target) - {p.name for p in providers_tgt}
        if skipped:
            print(f"  ⚠️  --target filter: names not found in configured providers: {skipped}")
    else:
        providers_tgt = providers

    if args.scenario:
        valid_names = {s.name for s in SCENARIOS}
        bad = set(args.scenario) - valid_names
        if bad:
            print(f"  ⚠️  --scenario filter: unknown scenario names: {bad}")
            print(f"       valid names: {sorted(valid_names)}")
        active_scenarios = [s for s in SCENARIOS if s.name in args.scenario]
    else:
        active_scenarios = list(SCENARIOS)

    # 1b. 按 CLI 参数过滤 stream modes
    if args.stream:
        valid_stream_names = {"stream", "nostream"}
        bad_stream = set(args.stream) - valid_stream_names
        if bad_stream:
            print(f"  ⚠️  --stream filter: unknown mode names: {bad_stream}")
            print(f"       valid names: {sorted(valid_stream_names)}")
        active_stream_modes: list[bool] = []
        if "nostream" in args.stream:
            active_stream_modes.append(False)
        if "stream" in args.stream:
            active_stream_modes.append(True)
    else:
        active_stream_modes = [False, True]

    if not providers_src or not providers_tgt or not active_scenarios or not active_stream_modes:
        print("ERROR: After filtering, no cases to run.")
        return 1

    case_count = len(active_scenarios) * len(providers_src) * len(providers_tgt) * len(active_stream_modes)
    stream_labels = ["stream" if s else "nostream" for s in active_stream_modes]
    print(
        f"\n  📋 Test scope: {len(active_scenarios)} scenarios × {len(providers_src)} sources"
        f" × {len(providers_tgt)} targets × {len(active_stream_modes)} stream modes = {case_count} cases"
    )
    print(f"     Scenarios: {[s.name for s in active_scenarios]}")
    print(f"     Sources:   {[p.name for p in providers_src]}")
    print(f"     Targets:   {[p.name for p in providers_tgt]}")
    print(f"     Stream:    {stream_labels}")

    # 2. 预检验活
    if args.skip_preflight:
        print("\n  ⏩ Preflight skipped (--skip-preflight)")
        dead_providers: set[str] = set()
    else:
        # 只验活实际用到的 provider（去重）
        providers_to_check = list({p.name: p for p in [*providers_src, *providers_tgt]}.values())
        health_results, _all_healthy = _run_preflight(providers_to_check, public_key, secret_key, host)
        dead_providers = {r.name for r in health_results if not r.alive and r.name != "langfuse"}
        langfuse_alive = next((r.alive for r in health_results if r.name == "langfuse"), False)
        if not langfuse_alive:
            print("\n  ⚠️  Langfuse is unreachable — traces will NOT be verified.\n")

    results: list[MatrixCaseResult] = []
    total = len(active_scenarios) * len(providers_src) * len(providers_tgt) * len(active_stream_modes)

    # 3. 记录测试起止时间，用于 Langfuse 时间范围查询
    run_started_at = datetime.now(UTC)
    print(f"\n  ⏱️  Run started: {run_started_at.isoformat()}")

    # 4. 加载 checkpoint（断点恢复）
    restored: dict[str, MatrixCaseResult] = {}
    if args.resume:
        restored = _load_checkpoint()
        if restored:
            print(f"\n  ♻️  Checkpoint loaded: {len(restored)} passed cases will be skipped")
            print(f"     Checkpoint file: {CHECKPOINT_FILE}")
        else:
            print("\n  ♻️  --resume specified but no valid checkpoint found, starting fresh")

    # 5. 分离 blocked / resumed / runnable cases
    runnable: list[tuple[ProviderConfig, ProviderConfig, ContentScenario, bool]] = []

    for scenario in active_scenarios:
        for source in providers_src:
            for target in providers_tgt:
                for stream_mode in active_stream_modes:
                    stream_tag = "stream" if stream_mode else "nostream"
                    key = _case_key(source.name, target.name, scenario.name, stream_mode)

                    # 5a. checkpoint 命中 → 跳过
                    if key in restored:
                        r = restored[key]
                        print(f"  ♻️  {source.name} → {target.name} | {scenario.name} | {stream_tag} — RESUMED (passed)")
                        results.append(r)
                        continue

                    # 5b. dead provider → blocked
                    blocked_by = {source.name, target.name} & dead_providers
                    if blocked_by:
                        print(f"  🚫 {source.name} → {target.name} | {scenario.name} | {stream_tag} — BLOCKED ({', '.join(blocked_by)})")
                        results.append(
                            MatrixCaseResult(
                                case_name=f"{source.name}_to_{target.name}__{scenario.name}__{stream_tag}",
                                source_provider=source.name,
                                target_provider=target.name,
                                source_api_type=source.api_type,
                                target_api_type=target.api_type,
                                scenario=scenario.name,
                                stream_mode=stream_mode,
                                session_id="",
                                status="blocked",
                                first_turn_ok=False,
                                second_turn_ok=False,
                                first_turn_excerpt="",
                                second_turn_excerpt="",
                                final_a_detected=False,
                                final_b_detected=False,
                                error=f"Provider(s) unreachable: {', '.join(blocked_by)}",
                                duration_seconds=0.0,
                            ),
                        )
                        continue

                    runnable.append((source, target, scenario, stream_mode))

    # 6. 串行执行 runnable cases（逐个运行，Ctrl+C 优雅停止）
    print(f"\n  🚀 Running {len(runnable)} cases  |  resumed={len(restored)}\n")

    _shutdown_requested = False
    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum: int, frame: object) -> None:
        nonlocal _shutdown_requested
        if _shutdown_requested:
            print("\n\n  ⚡ Force quit (2nd Ctrl+C). Checkpoint already saved.")
            signal.signal(signal.SIGINT, original_sigint)
            os._exit(1)
        _shutdown_requested = True
        print("\n\n  ⏸️  Graceful shutdown requested (Ctrl+C)...")
        print("     Current case will finish, then checkpoint is saved.")
        print("     Press Ctrl+C again to force quit.\n")

    signal.signal(signal.SIGINT, _sigint_handler)

    def _fmt_progress(idx: int, r: MatrixCaseResult) -> str:
        """Format a one-line progress string for a completed case."""
        icon = {"passed": "✅", "blocked": "🚫", "failed": "❌"}.get(r.status, "❓")
        t1 = "✓" if r.first_turn_ok else "✗"
        t2 = "✓" if r.second_turn_ok else "✗"
        s_tag = "stream" if r.stream_mode else "nostream"
        line = (
            f"  [{idx}/{total}] {icon} {r.source_provider} → {r.target_provider}"
            f" | {r.scenario} | {s_tag} | {r.status} ({r.duration_seconds:.1f}s) T1={t1} T2={t2}"
        )
        if r.error:
            line += f"  {r.error[:80]}"
        return line

    for src, tgt, sc, sm in runnable:
        if _shutdown_requested:
            # 剩余 case 标记为 blocked
            for remaining_src, remaining_tgt, remaining_sc, remaining_sm in runnable[runnable.index((src, tgt, sc, sm)) :]:
                remaining_stream_tag = "stream" if remaining_sm else "nostream"
                results.append(
                    MatrixCaseResult(
                        case_name=f"{remaining_src.name}_to_{remaining_tgt.name}__{remaining_sc.name}__{remaining_stream_tag}",
                        source_provider=remaining_src.name,
                        target_provider=remaining_tgt.name,
                        source_api_type=remaining_src.api_type,
                        target_api_type=remaining_tgt.api_type,
                        scenario=remaining_sc.name,
                        stream_mode=remaining_sm,
                        session_id="",
                        status="blocked",
                        first_turn_ok=False,
                        second_turn_ok=False,
                        first_turn_excerpt="",
                        second_turn_excerpt="",
                        final_a_detected=False,
                        final_b_detected=False,
                        error="Interrupted by Ctrl+C",
                        duration_seconds=0.0,
                    ),
                )
            _save_checkpoint(results)
            print(f"\n  💾 Checkpoint saved → {CHECKPOINT_FILE}")
            print("     Resume with: --resume --skip-preflight")
            break

        try:
            result = _run_case(src, tgt, sc, public_key, secret_key, host, stream=sm)
        except Exception as exc:
            sm_tag = "stream" if sm else "nostream"
            result = MatrixCaseResult(
                case_name=f"{src.name}_to_{tgt.name}__{sc.name}__{sm_tag}",
                source_provider=src.name,
                target_provider=tgt.name,
                source_api_type=src.api_type,
                target_api_type=tgt.api_type,
                scenario=sc.name,
                stream_mode=sm,
                session_id="",
                status="failed",
                first_turn_ok=False,
                second_turn_ok=False,
                first_turn_excerpt="",
                second_turn_excerpt="",
                final_a_detected=False,
                final_b_detected=False,
                error=f"[Unexpected] {type(exc).__name__}: {exc}",
                duration_seconds=0.0,
            )
        results.append(result)
        print(_fmt_progress(len(results), result))
        _save_checkpoint(results)

    # 恢复默认 SIGINT handler
    signal.signal(signal.SIGINT, original_sigint)

    # 8. 按 (scenario, source, target, stream_mode) 排序，确保报告顺序一致
    _scenario_order = {s.name: i for i, s in enumerate(SCENARIOS)}
    _provider_order = {p.name: i for i, p in enumerate(providers)}
    results.sort(
        key=lambda r: (
            _scenario_order.get(r.scenario, 999),
            _provider_order.get(r.source_provider, 999),
            _provider_order.get(r.target_provider, 999),
            0 if not r.stream_mode else 1,
        ),
    )

    # Write results
    Path(RESULTS_JSON).parent.mkdir(parents=True, exist_ok=True)

    serialized = [
        {
            "case_name": r.case_name,
            "source_provider": r.source_provider,
            "target_provider": r.target_provider,
            "scenario": r.scenario,
            "stream_mode": r.stream_mode,
            "session_id": r.session_id,
            "source_api_type": r.source_api_type,
            "target_api_type": r.target_api_type,
            "first_turn_ok": r.first_turn_ok,
            "second_turn_ok": r.second_turn_ok,
            "first_turn_excerpt": r.first_turn_excerpt,
            "second_turn_excerpt": r.second_turn_excerpt,
            "final_a_detected": r.final_a_detected,
            "final_b_detected": r.final_b_detected,
            "status": r.status,
            "error": r.error,
            "duration_seconds": r.duration_seconds,
        }
        for r in results
    ]

    report = _render_report(results)
    with open(RESULTS_REPORT, "w") as f:
        f.write(report)

    # Summary
    run_ended_at = datetime.now(UTC)
    run_duration = (run_ended_at - run_started_at).total_seconds()
    passed = sum(1 for r in results if r.status == "passed")
    blocked = sum(1 for r in results if r.status == "blocked")
    failed = sum(1 for r in results if r.status == "failed")

    # 时间窗口加缓冲：from -1s（避免边界），to +30s（Langfuse 异步写入延迟）
    from_ts = (run_started_at - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    to_ts = (run_ended_at + timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {passed} passed / {blocked} blocked / {failed} failed  (total={len(results)})")
    print(f"  Results → {RESULTS_JSON}")
    print(f"  Report  → {RESULTS_REPORT}")
    print(f"{'=' * 70}")
    print(f"  ⏱️  Started : {run_started_at.isoformat()}")
    print(f"  ⏱️  Ended   : {run_ended_at.isoformat()}")
    print(f"  ⏱️  Duration: {run_duration:.1f}s")

    # Per-scenario breakdown
    for scenario in active_scenarios:
        sc_results = [r for r in results if r.scenario == scenario.name]
        sc_passed = sum(1 for r in sc_results if r.status == "passed")
        sc_blocked = sum(1 for r in sc_results if r.status == "blocked")
        sc_failed = sum(1 for r in sc_results if r.status == "failed")
        print(f"    {scenario.name:25s}: {sc_passed} passed / {sc_blocked} blocked / {sc_failed} failed")

    # UMP coverage confirmation
    all_blocks: set[str] = set()
    for sc in active_scenarios:
        all_blocks.update(sc.ump_blocks_exercised)
    print(f"\n  UMP block types covered: {sorted(all_blocks)}")

    # ------------------------------------------------------------------
    # 失败诊断（按 provider pair × scenario 聚合）
    # ------------------------------------------------------------------
    non_passed = [r for r in results if r.status != "passed"]
    if non_passed:
        print(f"\n{'=' * 70}")
        print(f"  FAILURE DIAGNOSIS ({len(non_passed)} cases)")
        print(f"{'=' * 70}")

        for r in non_passed:
            icon = "🚫" if r.status == "blocked" else "❌"
            s_tag = "stream" if r.stream_mode else "nostream"
            print(f"\n  {icon} {r.source_provider} → {r.target_provider}  |  {r.scenario}  |  {s_tag}  |  {r.status}")
            print(f"     API: {r.source_api_type} → {r.target_api_type}")
            print(f"     T1={r.first_turn_ok}  T2={r.second_turn_ok}  ({r.duration_seconds:.1f}s)")
            if r.error:
                print(f"     Error: {r.error[:200]}")
            if r.first_turn_excerpt and not r.first_turn_ok:
                print(f"     T1 excerpt: {r.first_turn_excerpt[:150]}")
            if r.second_turn_excerpt and not r.second_turn_ok:
                print(f"     T2 excerpt: {r.second_turn_excerpt[:150]}")
            print(f"     Session: {r.session_id}")

        # 按 provider 聚合
        provider_fails: dict[str, int] = {}
        scenario_fails: dict[str, int] = {}
        for r in non_passed:
            src_key = f"{r.source_provider}({r.source_api_type})"
            tgt_key = f"{r.target_provider}({r.target_api_type})"
            provider_fails[src_key] = provider_fails.get(src_key, 0) + 1
            provider_fails[tgt_key] = provider_fails.get(tgt_key, 0) + 1
            scenario_fails[r.scenario] = scenario_fails.get(r.scenario, 0) + 1

        print(f"\n  {'─' * 60}")
        print("  📊 Failure heatmap:")
        print("     By provider (as source or target):")
        for prov, cnt in sorted(provider_fails.items(), key=lambda x: -x[1]):
            print(f"       {prov}: {cnt} failures")
        print("     By scenario:")
        for sc_name, cnt in sorted(scenario_fails.items(), key=lambda x: -x[1]):
            print(f"       {sc_name}: {cnt} failures")

    # ------------------------------------------------------------------
    # Langfuse 人工验证
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  LANGFUSE VERIFICATION")
    print(f"{'=' * 70}")

    # 1. 收集 session_ids 供人工验证
    passed_results = [r for r in results if r.status == "passed"]
    all_session_ids: list[str] = []
    for r in passed_results:
        if r.session_id:
            all_session_ids.append(r.session_id)

    # 2. 打印 curl 命令供人工验证
    print(f"\n  {'─' * 60}")
    print("  📋 Manual verification commands (copy & run):")
    print(f"  {'─' * 60}")

    # 2a. 查询本次运行所有 traces
    print("\n  # 查询本次运行所有 traces（按时间 + tag 过滤）")
    print(f"  curl -s -u '{public_key}:{secret_key}' \\")
    print(f"    '{host}/api/public/traces?tags=thinking-matrix-96&fromTimestamp={from_ts}&toTimestamp={to_ts}&limit=100' \\")
    print("    | python3 -m json.tool | head -50")

    # 2b. 查询单个 session 的详细 traces（取第一个 passed session 为例）
    if all_session_ids:
        sample_sid = all_session_ids[0]
        print(f"\n  # 查看单个 session 的 traces（示例: {sample_sid}）")
        print(f"  curl -s -u '{public_key}:{secret_key}' \\")
        print(f"    '{host}/api/public/traces?sessionId={sample_sid}' \\")
        print("    | python3 -m json.tool")

    # 3. 打印 Langfuse Web UI 链接
    base_host = host.rstrip("/")
    print(f"\n  {'─' * 60}")
    print("  🔗 Langfuse Web UI links:")
    print(f"  {'─' * 60}")
    print("\n  # 按 tag 筛选（在 Traces 页面搜索 thinking-matrix-96）")
    print(f"  {base_host}/traces?tags=thinking-matrix-96")

    if all_session_ids:
        print("\n  # 各 session 详情（每个 session = 一个 two-turn case）")
        for sid in all_session_ids[:8]:
            print(f"  {base_host}/sessions/{sid}")
        if len(all_session_ids) > 8:
            print(f"  ... and {len(all_session_ids) - 8} more sessions")

    # 4. 写入 JSON results 增加时间信息
    run_metadata = {
        "run_started_at": run_started_at.isoformat(),
        "run_ended_at": run_ended_at.isoformat(),
        "run_duration_seconds": round(run_duration, 2),
        "langfuse_host": host,
        "langfuse_tag": "thinking-matrix-96",
        "from_timestamp": from_ts,
        "to_timestamp": to_ts,
        "total_cases": len(results),
        "passed": passed,
        "blocked": blocked,
        "failed": failed,
        "session_ids": all_session_ids,
    }
    full_output = {"metadata": run_metadata, "cases": serialized}
    with open(RESULTS_JSON, "w") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")

    # Exit 0 when no real failures (blocked is OK)
    return_code = 1 if failed > 0 else 0
    return return_code


if __name__ == "__main__":
    sys.exit(main())
