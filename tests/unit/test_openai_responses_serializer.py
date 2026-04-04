from nexau.core.serializers.openai_responses import (
    normalize_openai_responses_api_tools,
    prepare_openai_responses_api_input,
    sanitize_openai_responses_items_for_input,
)


def test_openai_responses_serializer_prepares_reasoning_replay_and_instructions() -> None:
    prepared, instructions = prepare_openai_responses_api_input(
        [
            {"role": "system", "content": [{"type": "text", "text": "Be concise."}]},
            {
                "role": "assistant",
                "content": "Final: 34",
                "reasoning_content": "Compute 35 * 11 first, then divide by 13.",
            },
        ]
    )

    assert instructions == "Be concise."
    # RFC-0014: reasoning items 必须在 assistant message 之前，
    # 与 Responses API 原生输出顺序一致：reasoning → message → function_call
    assert prepared == [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Compute 35 * 11 first, then divide by 13."}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Final: 34"}],
        },
    ]


def test_openai_responses_serializer_sanitizes_multimodal_function_call_output() -> None:
    sanitized = sanitize_openai_responses_items_for_input(
        [
            {
                "id": "fc_out_1",
                "type": "function_call_output",
                "output": [
                    {"type": "output_text", "text": "caption"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg", "detail": "high"}},
                ],
            },
        ],
        drop_ephemeral_ids=True,
    )

    assert sanitized == [
        {
            "type": "function_call_output",
            "output": [
                {"type": "input_text", "text": "caption"},
                {"type": "input_image", "image_url": "https://example.com/a.jpg", "detail": "high"},
            ],
        },
    ]


def test_openai_responses_serializer_normalizes_function_tools() -> None:
    normalized = normalize_openai_responses_api_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "simple_tool",
                    "description": "A simple test tool",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": True,
                },
            },
        ]
    )

    assert normalized == [
        {
            "type": "function",
            "name": "simple_tool",
            "description": "A simple test tool",
            "parameters": {"type": "object", "properties": {}},
            "strict": True,
        },
    ]
