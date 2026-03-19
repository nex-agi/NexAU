from unittest.mock import patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.token_trace_session import TokenTraceContextOverflowError, TokenTraceSession
from nexau.core.messages import Message, Role, ToolResultBlock


def test_build_generate_with_token_kwargs_uses_input_ids_payload():
    llm_config = LLMConfig(
        model="token-model",
        base_url="http://token-gateway",
        api_key="test-key",
        api_type="generate_with_token",
    )
    session = TokenTraceSession(llm_config=llm_config, token_ids=[1, 2, 3])

    payload = session.build_generate_with_token_kwargs(
        max_output_tokens=32,
        request_params={
            "temperature": 0.2,
            "top_p": 0.9,
            "stop": ["</answer>"],
            "stream": False,
            "return_logprob": True,
            "sampling_params": {"skip_special_tokens": False},
        },
    )

    assert payload["model"] == "token-model"
    assert payload["input_ids"] == [1, 2, 3]
    assert payload["stream"] is False
    assert payload["return_logprob"] is True
    assert payload["sampling_params"]["max_new_tokens"] == 32
    assert payload["sampling_params"]["temperature"] == 0.2
    assert payload["sampling_params"]["top_p"] == 0.9
    assert payload["sampling_params"]["stop"] == ["</answer>"]
    assert payload["sampling_params"]["skip_special_tokens"] is False


def test_trace_session_accumulates_multi_turn_tokens():
    llm_config = LLMConfig(
        model="token-model",
        base_url="http://token-gateway",
        api_key="test-key",
        api_type="generate_with_token",
    )
    session = TokenTraceSession(llm_config=llm_config)

    tool_result_message = Message(
        role=Role.TOOL,
        content=[ToolResultBlock(tool_use_id="call_1", content='{"result": 4}')],
    )

    with patch.object(session, "tokenize_messages", side_effect=[[11, 12], [31, 32]]) as mock_tokenize:
        session.initialize_from_messages([Message.user("Hello")])
        session.append_model_response(
            output_token_ids=[21, 22, 23],
            fallback_messages=[Message.assistant("Calling tool")],
        )
        session.append_messages([tool_result_message], mask_value=0)
        session.append_model_response(
            output_token_ids=[41],
            fallback_messages=[Message.assistant("Done")],
        )

    trace = session.export_trace()

    assert trace["final_token_list"] == [11, 12, 21, 22, 23, 31, 32, 41]
    assert trace["response_mask"] == [0, 0, 1, 1, 1, 0, 0, 1]
    assert session.synced_message_count == 4
    assert mock_tokenize.call_count == 2
    assert mock_tokenize.call_args_list[0].kwargs["add_generation_prompt"] is True
    assert mock_tokenize.call_args_list[1].kwargs["add_generation_prompt"] is True


def test_initialize_from_messages_passes_tools_to_chat_template():
    llm_config = LLMConfig(
        model="token-model",
        base_url="http://token-gateway",
        api_key="test-key",
        api_type="generate_with_token",
    )
    session = TokenTraceSession(llm_config=llm_config)
    tools = [
        {
            "name": "search",
            "description": "Search the web.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            "kind": "tool",
        }
    ]

    with patch.object(session, "_get_hf_tokenizer") as mock_get_tokenizer:
        mock_tokenizer = mock_get_tokenizer.return_value
        mock_tokenizer.apply_chat_template.return_value = {"input_ids": [11, 12]}

        session.initialize_from_messages([Message.user("Hello")], tools=tools)

    mock_tokenizer.apply_chat_template.assert_called_once_with(
        conversation=[{"role": "user", "content": "Hello"}],
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum results",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
    )
    assert session.token_ids == [11, 12]
    assert session.response_mask == [0, 0]


def test_append_model_response_fallback_does_not_add_generation_prompt():
    llm_config = LLMConfig(
        model="token-model",
        base_url="http://token-gateway",
        api_key="test-key",
        api_type="generate_with_token",
    )
    session = TokenTraceSession(llm_config=llm_config)

    with patch.object(session, "tokenize_messages", return_value=[21, 22]) as mock_tokenize:
        session.append_model_response(
            output_token_ids=[],
            fallback_messages=[Message.assistant("Done")],
        )

    mock_tokenize.assert_called_once()
    called_messages = mock_tokenize.call_args.args[0]
    assert len(called_messages) == 1
    assert called_messages[0].role == Role.ASSISTANT
    assert called_messages[0].get_text_content() == "Done"
    assert mock_tokenize.call_args.kwargs == {}


def test_sync_external_messages_adds_generation_prompt():
    llm_config = LLMConfig(
        model="token-model",
        base_url="http://token-gateway",
        api_key="test-key",
        api_type="generate_with_token",
    )
    session = TokenTraceSession(llm_config=llm_config, synced_message_count=1)
    trailing_messages = [Message.user("Hello"), Message.user("Continue")]

    with patch.object(session, "tokenize_messages", return_value=[31, 32]) as mock_tokenize:
        session.sync_external_messages(trailing_messages)

    mock_tokenize.assert_called_once()
    called_messages = mock_tokenize.call_args.args[0]
    assert len(called_messages) == 1
    assert called_messages[0].role == Role.USER
    assert called_messages[0].get_text_content() == "Continue"
    assert mock_tokenize.call_args.kwargs == {"add_generation_prompt": True}
    assert session.token_ids == [31, 32]
    assert session.response_mask == [0, 0]
    assert session.synced_message_count == 2


def test_sync_external_messages_flushes_buffered_tool_messages():
    llm_config = LLMConfig(
        model="token-model",
        base_url="http://token-gateway",
        api_key="test-key",
        api_type="generate_with_token",
    )
    session = TokenTraceSession(llm_config=llm_config, synced_message_count=2)
    tool_result_message = Message(
        role=Role.TOOL,
        content=[ToolResultBlock(tool_use_id="call_1", content='{"result": 4}')],
    )
    full_messages = [Message.user("Hello"), Message.assistant("Calling tool"), tool_result_message]

    with patch.object(session, "tokenize_messages", return_value=[31, 32]) as mock_tokenize:
        buffered_tokens = session.append_messages([tool_result_message], mask_value=0)
        session.sync_external_messages(full_messages)

    assert buffered_tokens == []
    mock_tokenize.assert_called_once_with([tool_result_message], add_generation_prompt=True)
    assert session.token_ids == [31, 32]
    assert session.response_mask == [0, 0]
    assert session.synced_message_count == 3


def test_sync_external_messages_raises_on_assistant_messages():
    llm_config = LLMConfig(
        model="token-model",
        base_url="http://token-gateway",
        api_key="test-key",
        api_type="generate_with_token",
    )
    session = TokenTraceSession(llm_config=llm_config, synced_message_count=1)

    trailing_messages = [
        Message.user("Hello"),
        Message.assistant("Internal draft"),
        Message.user("Continue"),
    ]

    with patch.object(session, "tokenize_messages") as mock_tokenize:
        with pytest.raises(ValueError, match="does not accept assistant messages"):
            session.sync_external_messages(trailing_messages)

    mock_tokenize.assert_not_called()
    assert session.token_ids == []
    assert session.response_mask == []
    assert session.synced_message_count == 1


def test_append_messages_batches_sequential_tool_messages_until_boundary():
    llm_config = LLMConfig(
        model="token-model",
        base_url="http://token-gateway",
        api_key="test-key",
        api_type="generate_with_token",
    )
    session = TokenTraceSession(llm_config=llm_config)
    first_tool_result = Message(
        role=Role.TOOL,
        content=[ToolResultBlock(tool_use_id="call_1", content='{"result": 4}')],
    )
    second_tool_result = Message(
        role=Role.TOOL,
        content=[ToolResultBlock(tool_use_id="call_2", content='{"result": 5}')],
    )

    with patch.object(session, "tokenize_messages", side_effect=[[11, 12], [31, 32, 33]]) as mock_tokenize:
        session.initialize_from_messages([Message.user("Hello")])
        first_tokens = session.append_messages([first_tool_result], mask_value=0)
        second_tokens = session.append_messages([second_tool_result], mask_value=0)
        session.append_model_response(
            output_token_ids=[41],
            fallback_messages=[Message.assistant("Done")],
        )

    assert first_tokens == []
    assert second_tokens == []
    assert mock_tokenize.call_count == 2
    buffered_messages = mock_tokenize.call_args_list[1].args[0]
    assert [message.role for message in buffered_messages] == [Role.TOOL, Role.TOOL]
    assert session.token_ids == [11, 12, 31, 32, 33, 41]
    assert session.response_mask == [0, 0, 0, 0, 0, 1]
    assert session.synced_message_count == 4


def test_overflow_raises_when_exceeding_max_context_tokens():
    """超过 max_context_tokens 时，_append_tokens 直接抛出 TokenTraceContextOverflowError。"""
    llm_config = LLMConfig(
        model="token-model",
        base_url="http://token-gateway",
        api_key="test-key",
        api_type="generate_with_token",
    )
    session = TokenTraceSession(llm_config=llm_config, max_context_tokens=10)

    with patch.object(session, "tokenize_messages", return_value=[1, 2, 3, 4, 5]):
        session.initialize_from_messages([Message.user("Hello")])

    assert len(session.token_ids) == 5

    # 追加 model response（5 tokens），总计 10 tokens，刚好不超限
    session.append_model_response(
        output_token_ids=[6, 7, 8, 9, 10],
        fallback_messages=[Message.assistant("ok")],
    )
    assert len(session.token_ids) == 10

    # 再追加 1 个 token 应触发 overflow
    with pytest.raises(TokenTraceContextOverflowError, match="exceeding max_context_tokens"):
        session.append_model_response(
            output_token_ids=[11],
            fallback_messages=[Message.assistant("boom")],
        )

    # token buffer 应保持不变（未追加）
    assert len(session.token_ids) == 10


def test_no_overflow_when_max_context_tokens_is_none():
    """max_context_tokens 为 None 时不做溢出检查。"""
    llm_config = LLMConfig(
        model="token-model",
        base_url="http://token-gateway",
        api_key="test-key",
        api_type="generate_with_token",
    )
    session = TokenTraceSession(llm_config=llm_config, max_context_tokens=None)

    session._append_tokens(list(range(1000)), mask_value=0)
    assert len(session.token_ids) == 1000


def test_overflow_on_append_messages():
    """append_messages 路径同样触发溢出检查。"""
    llm_config = LLMConfig(
        model="token-model",
        base_url="http://token-gateway",
        api_key="test-key",
        api_type="generate_with_token",
    )
    session = TokenTraceSession(llm_config=llm_config, max_context_tokens=5)

    with patch.object(session, "tokenize_messages", return_value=[1, 2, 3]):
        session.initialize_from_messages([Message.user("Hello")])

    tool_result_message = Message(
        role=Role.TOOL,
        content=[ToolResultBlock(tool_use_id="call_1", content='{"result": 4}')],
    )
    with patch.object(session, "tokenize_messages", return_value=[4, 5, 6]):
        with pytest.raises(TokenTraceContextOverflowError):
            session.append_messages([tool_result_message], mask_value=0)
