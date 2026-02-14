# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for web_fetch builtin tool."""

from nexau.archs.tool.builtin.web_tools import web_fetch


class TestWebFetch:
    """Test web_fetch tool functionality."""

    def test_error_when_prompt_empty(self):
        """Should return error when prompt is empty."""
        result = web_fetch(prompt="")

        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PROMPT"

    def test_error_when_no_urls_in_prompt(self):
        """Should return error when prompt contains no valid URLs."""
        result = web_fetch(prompt="Please analyze this text without any URLs")

        assert result.get("error") is not None
        assert result["error"]["type"] == "NO_URLS_FOUND"

    def test_error_when_url_invalid_protocol(self):
        """Should return error for unsupported protocols."""
        result = web_fetch(prompt="Fetch content from ftp://example.com/file")

        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_URL"
        assert "unsupported" in result["content"].lower()


class TestWebFetchURLParsing:
    """Test web_fetch URL parsing from prompts."""

    def test_url_param_maps_to_prompt(self):
        """Should accept url param and use it as prompt."""
        result = web_fetch(url="https://example.com")
        if result.get("error"):
            assert result["error"]["type"] not in ["INVALID_PROMPT", "NO_URLS_FOUND"]

    def test_extract_https_url(self):
        """Should extract https:// URL from prompt."""
        result = web_fetch(prompt="Analyze this page https://example.com and summarize")

        if result.get("error"):
            assert result["error"]["type"] not in ["INVALID_PROMPT", "NO_URLS_FOUND"]


class TestWebFetchOutputFormat:
    """Test web_fetch output format."""

    def test_error_format(self):
        """Should format error correctly."""
        result = web_fetch(prompt="")

        assert result.get("error") is not None
        assert "message" in result["error"]
        assert "type" in result["error"]

    def test_llm_content_format_error(self):
        """Should format llmContent correctly on error."""
        result = web_fetch(prompt="")

        assert "empty" in result["content"].lower() or "prompt" in result["content"].lower()

    def test_return_display_format_error(self):
        """Should format returnDisplay correctly on error."""
        result = web_fetch(prompt="")

        assert "Error" in result["returnDisplay"]
