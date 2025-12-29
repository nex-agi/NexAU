import json

from nexau.core.messages import ImageBlock, TextBlock, coerce_tool_result_content


def test_coerce_tool_result_content_list_hits_string_branch_and_drops_empty_strings() -> None:
    # Non-string top-level input triggers coerce_any(), including the `isinstance(item, str)` branch.
    coerced = coerce_tool_result_content(["hello", ""])
    assert coerced == "hello"


def test_coerce_tool_result_content_none_returns_empty_string() -> None:
    assert coerce_tool_result_content(None) == ""


def test_coerce_tool_result_content_json_string_with_image_returns_blocks() -> None:
    payload = [
        {"type": "output_text", "text": "hi"},
        {"type": "input_image", "image_url": "https://example.com/a.png", "detail": "low"},
    ]
    coerced = coerce_tool_result_content(json.dumps(payload))
    assert isinstance(coerced, list)
    assert [b.type for b in coerced] == ["text", "image"]
    text = next(b for b in coerced if isinstance(b, TextBlock))
    image = next(b for b in coerced if isinstance(b, ImageBlock))
    assert text.text == "hi"
    assert image.url == "https://example.com/a.png"
    assert image.detail == "low"


def test_coerce_tool_result_content_json_string_without_image_uses_fallback_text() -> None:
    payload = [{"type": "output_text", "text": "hi"}]
    coerced = coerce_tool_result_content(json.dumps(payload), fallback_text="FALLBACK")
    assert coerced == "FALLBACK"


def test_coerce_tool_result_content_invalid_json_string_falls_back() -> None:
    raw = "{not valid json"
    assert coerce_tool_result_content(raw, fallback_text="FALLBACK") == "FALLBACK"
    assert coerce_tool_result_content(raw) == raw


def test_coerce_tool_result_content_empty_string_falls_back() -> None:
    assert coerce_tool_result_content("", fallback_text="FALLBACK") == "FALLBACK"
    assert coerce_tool_result_content("") == ""
