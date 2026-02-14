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

"""Unit tests for google_web_search builtin tool."""

from unittest.mock import patch

from nexau.archs.tool.builtin.web_tools import google_web_search


class TestGoogleWebSearch:
    """Test google_web_search tool functionality."""

    def test_error_when_query_empty(self):
        """Should return error when query is empty."""
        result = google_web_search(query="")

        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_QUERY"

    def test_error_when_query_whitespace_only(self):
        """Should return error when query is only whitespace."""
        result = google_web_search(query="   ")

        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_QUERY"

    @patch("nexau.archs.tool.builtin.web_tools.google_web_search._web_search")
    def test_search_returns_error_when_serper_fails(self, mock_web_search):
        """Should return error when web_search returns error."""
        mock_web_search.return_value = {
            "status": "error",
            "error": "SERPER_API_KEY required",
        }
        result = google_web_search(query="test query")

        assert result.get("error") is not None
        assert result["error"]["type"] == "WEB_SEARCH_FAILED"


class TestGoogleWebSearchOutputFormat:
    """Test google_web_search output format."""

    def test_error_format(self):
        """Should format error correctly."""
        result = google_web_search(query="")

        assert result.get("error") is not None
        assert "message" in result["error"]
        assert "type" in result["error"]

    def test_llm_content_format_error(self):
        """Should format llmContent correctly on error."""
        result = google_web_search(query="")

        assert "query" in result["content"].lower() or "empty" in result["content"].lower()

    def test_return_display_format_error(self):
        """Should format returnDisplay correctly on error."""
        result = google_web_search(query="")

        display = result["returnDisplay"]
        assert "Error" in display or "empty" in display.lower()


class TestGoogleWebSearchWithMockedWebSearch:
    """Test google_web_search with mocked web_tool.web_search."""

    @patch("nexau.archs.tool.builtin.web_tools.google_web_search._web_search")
    def test_successful_search(self, mock_web_search):
        """Should format Serper results correctly."""
        mock_web_search.return_value = {
            "status": "success",
            "results": [
                {"title": "Python.org", "link": "https://python.org", "snippet": "Python is a programming language."},
            ],
        }
        result = google_web_search(query="python")

        assert result.get("error") is None
        assert "Python" in result["content"]
        assert "content" in result
        assert "returnDisplay" in result

    @patch("nexau.archs.tool.builtin.web_tools.google_web_search._web_search")
    def test_search_with_sources(self, mock_web_search):
        """Should include sources in result."""
        mock_web_search.return_value = {
            "status": "success",
            "results": [
                {"title": "Python.org", "link": "https://python.org"},
            ],
        }
        result = google_web_search(query="python")

        assert result.get("error") is None
        assert "Sources" not in result["content"]  # Serper format uses [1] Title (link)
        assert "Python.org" in result["content"]
        assert result.get("sources") is not None

    @patch("nexau.archs.tool.builtin.web_tools.google_web_search._web_search")
    def test_search_raises_exception(self, mock_web_search):
        """Should handle web_search exceptions gracefully."""
        mock_web_search.side_effect = Exception("Network error")

        result = google_web_search(query="test")

        assert result.get("error") is not None
        assert result["error"]["type"] == "WEB_SEARCH_FAILED"

    @patch("nexau.archs.tool.builtin.web_tools.google_web_search._web_search")
    def test_empty_search_results(self, mock_web_search):
        """Should handle empty search results."""
        mock_web_search.return_value = {"status": "success", "results": []}

        result = google_web_search(query="obscure query")

        assert "No results" in result["content"] or "no" in result["content"].lower()
