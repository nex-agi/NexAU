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

"""Unit tests for read_file and read_visual_file builtin tools."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from nexau.archs.sandbox import SandboxStatus
from nexau.archs.tool.builtin.file_tools import read_file, read_visual_file
from nexau.archs.tool.builtin.file_tools.read_file import (
    MAX_TOTAL_OUTPUT_CHARS,
    _detect_encoding,
    _head_truncate_lines,
    _read_text_lossy,
)


def _make_agent_state(sandbox):
    """Create mock agent_state with sandbox."""
    agent_state = Mock()
    agent_state.get_sandbox.return_value = sandbox
    return agent_state


def _make_sandbox_for_encoding(raw_bytes: bytes) -> Mock:
    """Create mock sandbox whose read_file(binary=True) returns raw_bytes."""
    sandbox = Mock()
    bin_res = Mock()
    bin_res.status = SandboxStatus.SUCCESS
    bin_res.content = raw_bytes
    sandbox.read_file.return_value = bin_res
    return sandbox


class TestDetectEncoding:
    """Test _detect_encoding layered strategy."""

    def test_pure_ascii_returns_utf8(self):
        """Pure ASCII content should return utf-8 (ASCII is a subset of UTF-8)."""
        raw = b"def hello():\n    print('world')\n"
        sandbox = _make_sandbox_for_encoding(raw)
        assert _detect_encoding("test.py", sandbox) == "utf-8"

    def test_ascii_head_chinese_tail_returns_utf8(self):
        """File with ASCII head and Chinese tail — the original bug scenario."""
        ascii_head = b"x = 1\n" * 2000
        chinese_tail = "# 这是中文注释\n".encode("utf-8")
        raw = ascii_head + chinese_tail
        assert len(ascii_head) > 10000
        sandbox = _make_sandbox_for_encoding(raw)
        assert _detect_encoding("verify.py", sandbox) == "utf-8"

    def test_utf8_with_multibyte_returns_utf8(self):
        """UTF-8 file with multi-byte characters should decode correctly."""
        raw = "こんにちは世界\n你好世界\n".encode("utf-8")
        sandbox = _make_sandbox_for_encoding(raw)
        assert _detect_encoding("test.txt", sandbox) == "utf-8"

    def test_utf8_bom_returns_utf8_sig(self):
        """File with UTF-8 BOM should return utf-8-sig."""
        raw = b"\xef\xbb\xbf" + b"hello"
        sandbox = _make_sandbox_for_encoding(raw)
        assert _detect_encoding("test.txt", sandbox) == "utf-8-sig"

    def test_utf16_le_bom_returns_utf16_le(self):
        """File with UTF-16 LE BOM should return utf-16-le."""
        raw = b"\xff\xfe" + "hello".encode("utf-16-le")
        sandbox = _make_sandbox_for_encoding(raw)
        assert _detect_encoding("test.txt", sandbox) == "utf-16-le"

    def test_utf16_be_bom_returns_utf16_be(self):
        """File with UTF-16 BE BOM should return utf-16-be."""
        raw = b"\xfe\xff" + "hello".encode("utf-16-be")
        sandbox = _make_sandbox_for_encoding(raw)
        assert _detect_encoding("test.txt", sandbox) == "utf-16-be"

    def test_utf32_le_bom_returns_utf32_le(self):
        """UTF-32 LE BOM should be detected before UTF-16 LE (prefix overlap)."""
        raw = b"\xff\xfe\x00\x00" + "hi".encode("utf-32-le")
        sandbox = _make_sandbox_for_encoding(raw)
        assert _detect_encoding("test.txt", sandbox) == "utf-32-le"

    def test_empty_content_returns_utf8(self):
        """Empty file content should fall back to utf-8."""
        sandbox = _make_sandbox_for_encoding(b"")
        assert _detect_encoding("test.txt", sandbox) == "utf-8"

    def test_binary_read_failure_returns_utf8(self):
        """If binary read fails, should fall back to utf-8."""
        sandbox = Mock()
        fail_res = Mock()
        fail_res.status = SandboxStatus.ERROR
        fail_res.content = None
        sandbox.read_file.return_value = fail_res
        assert _detect_encoding("test.txt", sandbox) == "utf-8"

    def test_non_utf8_falls_through_to_chardet(self, monkeypatch: pytest.MonkeyPatch):
        """Non-UTF-8 bytes should skip UTF-8 validation and adopt chardet result."""
        raw = b"\xc4\xe3\xba\xc3"  # "你好" in GBK, invalid UTF-8
        sandbox = _make_sandbox_for_encoding(raw)

        fake_chardet = Mock()
        fake_chardet.detect.return_value = {"encoding": "GB2312", "confidence": 0.9}
        monkeypatch.setitem(__import__("sys").modules, "chardet", fake_chardet)

        assert _detect_encoding("test.txt", sandbox) == "gb2312"


class TestReadFileEncodingFallback:
    """Test read_file falls back gracefully when encoding detection is wrong."""

    def test_fallback_on_decode_error(self):
        """If text read fails (e.g. wrong encoding), should fallback to binary + utf-8 replace."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        raw = b"hello \x80\x81\x82 world"

        def read_file_side_effect(path: str, encoding: str = "utf-8", binary: bool = False):
            res = Mock()
            if binary:
                res.status = SandboxStatus.SUCCESS
                res.content = raw
            else:
                res.status = SandboxStatus.ERROR
                res.error = "UnicodeDecodeError"
                res.content = None
            return res

        sandbox.read_file.side_effect = read_file_side_effect

        agent_state = _make_agent_state(sandbox)
        result = read_file(file_path="test.txt", agent_state=agent_state)

        assert "error" not in result or result.get("error") is None
        assert "hello" in result["content"]
        assert "world" in result["content"]


class TestReadFileLineLengthTruncation:
    """Test read_file single-line length truncation (MAX_LINE_LENGTH = 2000)."""

    def test_truncates_line_longer_than_2000_chars(self):
        """Should truncate lines exceeding 2000 chars and append '... [truncated]'."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        # Create content with one line > 2000 chars
        long_line = "a" * 2500
        content = f"short line\n{long_line}\nanother short"
        read_res = Mock()
        read_res.status = SandboxStatus.SUCCESS
        read_res.content = content
        read_res.truncated = False

        def read_file_side_effect(path, encoding=None, binary=False):
            if binary:
                return Mock(status=SandboxStatus.SUCCESS, content=b"")
            return read_res

        sandbox.read_file.side_effect = read_file_side_effect

        agent_state = _make_agent_state(sandbox)
        result = read_file(file_path="test.txt", agent_state=agent_state)

        assert "error" not in result or result.get("error") is None
        content = result["content"]
        # Truncated line should be 2000 + suffix, not 2500
        assert "... [truncated]" in content
        # Should not contain the full 2500-char line
        assert "a" * 2500 not in content
        # Should contain truncated version
        assert "a" * 2000 + "... [truncated]" in content
        assert "some lines were shortened" in result["returnDisplay"]

    def test_no_truncation_when_all_lines_under_limit(self):
        """Should not truncate when all lines are under 2000 chars."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        content = "line1\nline2\nline3"
        read_res = Mock()
        read_res.status = SandboxStatus.SUCCESS
        read_res.content = content
        read_res.truncated = False

        def read_file_side_effect(path, encoding=None, binary=False):
            if binary:
                return Mock(status=SandboxStatus.SUCCESS, content=b"")
            return read_res

        sandbox.read_file.side_effect = read_file_side_effect

        agent_state = _make_agent_state(sandbox)
        result = read_file(file_path="test.txt", agent_state=agent_state)

        assert "... [truncated]" not in result["content"]
        assert "some lines were shortened" not in result["returnDisplay"]


class TestReadFileRejectsVisualFiles:
    """Test that read_file rejects image/video files and directs to read_visual_file."""

    def test_image_file_returns_error(self):
        """Image files should return an error directing to read_visual_file."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        agent_state = _make_agent_state(sandbox)
        result = read_file(file_path="x.png", agent_state=agent_state)

        assert "error" in result
        assert result["error"]["type"] == "USE_READ_VISUAL_FILE"
        assert "read_visual_file" in result["content"]

    def test_video_file_returns_error(self):
        """Video files should return an error directing to read_visual_file."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        agent_state = _make_agent_state(sandbox)
        result = read_file(file_path="clip.mp4", agent_state=agent_state)

        assert "error" in result
        assert result["error"]["type"] == "USE_READ_VISUAL_FILE"
        assert "read_visual_file" in result["content"]


class TestReadVisualFileImage:
    """Test read_visual_file returns correct image format."""

    def test_image_returns_result_with_image_block_format(self):
        """Image files should return result with type=image, image_url=data:... for coerce."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        read_res = Mock()
        read_res.status = SandboxStatus.SUCCESS
        read_res.content = png_bytes

        def read_file_side_effect(path, encoding=None, binary=False):
            if binary:
                return read_res
            return Mock(status=SandboxStatus.SUCCESS, content="", truncated=False)

        sandbox.read_file.side_effect = read_file_side_effect

        agent_state = _make_agent_state(sandbox)
        result = read_visual_file(file_path="x.png", agent_state=agent_state)

        assert "error" not in result or result.get("error") is None
        assert "content" in result
        block = result["content"]
        assert block["type"] == "image"
        assert "image_url" in block
        assert block["image_url"].startswith("data:image/")
        assert "base64," in block["image_url"]
        assert block["detail"] == "auto"


class TestReadVisualFileRejectsTextFiles:
    """Test that read_visual_file rejects non-visual files."""

    def test_text_file_returns_error(self):
        """Text files should return an error directing to read_file."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        agent_state = _make_agent_state(sandbox)
        result = read_visual_file(file_path="test.txt", agent_state=agent_state)

        assert "error" in result
        assert result["error"]["type"] == "NOT_VISUAL_FILE"
        assert "read_file" in result["content"]


class TestReadVisualFileParameterSanitization:
    """Test that read_visual_file rejects malicious / invalid numeric parameters."""

    def test_rejects_string_injection_in_video_frame_interval(self):
        """A string like 'drawtext=...' in video_frame_interval should be rejected."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        agent_state = _make_agent_state(sandbox)
        result = read_visual_file(
            file_path="clip.mp4",
            video_frame_interval="1,drawtext=textfile=/etc/passwd",
            agent_state=agent_state,
        )

        assert "error" in result
        assert result["error"]["type"] == "INVALID_PARAMETER"

    def test_rejects_string_injection_in_video_frame_width(self):
        """A malicious string in video_frame_width should be rejected."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        agent_state = _make_agent_state(sandbox)
        result = read_visual_file(
            file_path="clip.mp4",
            video_frame_width="640,drawtext=text=pwned",
            agent_state=agent_state,
        )

        assert "error" in result
        assert result["error"]["type"] == "INVALID_PARAMETER"

    def test_rejects_string_injection_in_video_max_frames(self):
        """A malicious string in video_max_frames should be rejected."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        agent_state = _make_agent_state(sandbox)
        result = read_visual_file(
            file_path="clip.mp4",
            video_max_frames="5;rm -rf /",
            agent_state=agent_state,
        )

        assert "error" in result
        assert result["error"]["type"] == "INVALID_PARAMETER"

    def test_rejects_string_injection_in_image_max_size(self):
        """A malicious string in image_max_size should be rejected."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        agent_state = _make_agent_state(sandbox)
        result = read_visual_file(
            file_path="img.png",
            image_max_size="800,drawtext=textfile=/etc/shadow",
            agent_state=agent_state,
        )

        assert "error" in result
        assert result["error"]["type"] == "INVALID_PARAMETER"

    def test_rejects_invalid_image_detail(self):
        """image_detail must be 'low', 'high', or 'auto'."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        agent_state = _make_agent_state(sandbox)
        result = read_visual_file(
            file_path="img.png",
            image_detail="malicious_value",
            agent_state=agent_state,
        )

        assert "error" in result
        assert result["error"]["type"] == "INVALID_PARAMETER"

    def test_accepts_valid_int_castable_string_params(self):
        """Numeric strings like '5' should be accepted and cast to int."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        read_res = Mock()
        read_res.status = SandboxStatus.SUCCESS
        read_res.content = png_bytes

        def read_file_side_effect(path, encoding=None, binary=False):
            if binary:
                return read_res
            return Mock(status=SandboxStatus.SUCCESS, content="", truncated=False)

        sandbox.read_file.side_effect = read_file_side_effect

        agent_state = _make_agent_state(sandbox)
        # Pass castable string "800" as image_max_size — should not error at validation
        result = read_visual_file(
            file_path="img.png",
            image_max_size="800",
            agent_state=agent_state,
        )

        # Should succeed (no INVALID_PARAMETER error)
        assert result.get("error") is None or result["error"]["type"] != "INVALID_PARAMETER"


# ---------------------------------------------------------------------------
# Helpers for encoding tests
# ---------------------------------------------------------------------------


def _make_sandbox_with_bytes(raw: bytes) -> Mock:
    """Create a mock sandbox that returns *raw* bytes on binary read."""
    sandbox = Mock()
    sandbox.work_dir = Path("/tmp/work")
    sandbox.file_exists.return_value = True

    info = Mock()
    info.is_directory = False
    info.size = len(raw)
    sandbox.get_file_info.return_value = info

    read_res = Mock()
    read_res.status = SandboxStatus.SUCCESS
    read_res.content = raw

    sandbox.read_file.return_value = read_res
    return sandbox


def _call_read_file_with_bytes(raw: bytes, filename: str = "test.txt") -> dict:
    """Shortcut: build sandbox mock from raw bytes and call read_file end-to-end."""
    sandbox = _make_sandbox_with_bytes(raw)
    agent_state = _make_agent_state(sandbox)
    return read_file(file_path=filename, agent_state=agent_state)


# ---------------------------------------------------------------------------
# _read_text_lossy unit tests
# ---------------------------------------------------------------------------


class TestReadTextLossy:
    """Direct tests for the _read_text_lossy helper."""

    def test_pure_ascii(self):
        raw = b"hello world\nline two\n"
        sandbox = _make_sandbox_with_bytes(raw)
        assert _read_text_lossy("/f", sandbox) == "hello world\nline two\n"

    def test_utf8_chinese(self):
        raw = "你好世界\n第二行\n".encode("utf-8")
        sandbox = _make_sandbox_with_bytes(raw)
        assert _read_text_lossy("/f", sandbox) == "你好世界\n第二行\n"

    def test_utf8_with_bom(self):
        raw = b"\xef\xbb\xbf" + "BOM头文件\n".encode("utf-8")
        sandbox = _make_sandbox_with_bytes(raw)
        result = _read_text_lossy("/f", sandbox)
        assert "BOM头文件" in result

    def test_utf8_emoji(self):
        raw = "hello 🚀🎉\nworld 🌍\n".encode("utf-8")
        sandbox = _make_sandbox_with_bytes(raw)
        result = _read_text_lossy("/f", sandbox)
        assert "🚀" in result
        assert "🌍" in result

    def test_utf8_japanese(self):
        raw = "日本語テスト\n二行目\n".encode("utf-8")
        sandbox = _make_sandbox_with_bytes(raw)
        assert _read_text_lossy("/f", sandbox) == "日本語テスト\n二行目\n"

    def test_utf8_korean(self):
        raw = "한국어 테스트\n두 번째 줄\n".encode("utf-8")
        sandbox = _make_sandbox_with_bytes(raw)
        assert _read_text_lossy("/f", sandbox) == "한국어 테스트\n두 번째 줄\n"

    def test_utf8_mixed_scripts(self):
        raw = "English 中文 日本語 한국어 العربية\n".encode("utf-8")
        sandbox = _make_sandbox_with_bytes(raw)
        result = _read_text_lossy("/f", sandbox)
        assert "中文" in result
        assert "العربية" in result

    # -- non-UTF-8 encodings: invalid bytes become U+FFFD --

    def test_latin1_bytes_replaced(self):
        """Latin-1 bytes outside ASCII are not valid UTF-8 — should become U+FFFD."""
        raw = "caf\xe9\n\xfc\xf6\xe4\n".encode("latin-1")
        sandbox = _make_sandbox_with_bytes(raw)
        result = _read_text_lossy("/f", sandbox)
        assert "\ufffd" in result
        assert "caf" in result

    def test_gbk_chinese_replaced(self):
        """GBK-encoded Chinese is not valid UTF-8 — invalid bytes become U+FFFD."""
        raw = "你好世界".encode("gbk")
        sandbox = _make_sandbox_with_bytes(raw)
        result = _read_text_lossy("/f", sandbox)
        assert "\ufffd" in result
        # Should NOT decode correctly (GBK ≠ UTF-8)
        assert result != "你好世界"

    def test_shift_jis_replaced(self):
        """Shift-JIS encoded Japanese is not valid UTF-8."""
        raw = "日本語テスト".encode("shift_jis")
        sandbox = _make_sandbox_with_bytes(raw)
        result = _read_text_lossy("/f", sandbox)
        assert "\ufffd" in result

    def test_gb2312_replaced(self):
        """GB2312-encoded text is not valid UTF-8."""
        raw = "测试文本".encode("gb2312")
        sandbox = _make_sandbox_with_bytes(raw)
        result = _read_text_lossy("/f", sandbox)
        assert "\ufffd" in result

    def test_euc_kr_replaced(self):
        """EUC-KR encoded Korean is not valid UTF-8."""
        raw = "한국어".encode("euc_kr")
        sandbox = _make_sandbox_with_bytes(raw)
        result = _read_text_lossy("/f", sandbox)
        assert "\ufffd" in result

    def test_raw_invalid_bytes(self):
        """Completely invalid byte sequences → all become U+FFFD."""
        raw = b"\xff\xfe\x80\x81\x90\x91"
        sandbox = _make_sandbox_with_bytes(raw)
        result = _read_text_lossy("/f", sandbox)
        assert all(ch == "\ufffd" for ch in result)

    def test_mixed_valid_and_invalid_bytes(self):
        """Valid ASCII mixed with invalid bytes — valid parts preserved."""
        raw = b"hello\xff\xfeworld\x80end"
        sandbox = _make_sandbox_with_bytes(raw)
        result = _read_text_lossy("/f", sandbox)
        assert result.startswith("hello")
        assert "world" in result
        assert result.endswith("end")
        assert "\ufffd" in result

    def test_empty_file(self):
        sandbox = _make_sandbox_with_bytes(b"")
        result = _read_text_lossy("/f", sandbox)
        assert result == ""

    def test_sandbox_returns_str_passthrough(self):
        """When sandbox already returns str (remote sandbox), pass through as-is."""
        sandbox = Mock()
        read_res = Mock()
        read_res.status = SandboxStatus.SUCCESS
        read_res.content = "already decoded 已解码"
        sandbox.read_file.return_value = read_res
        result = _read_text_lossy("/f", sandbox)
        assert result == "already decoded 已解码"

    def test_sandbox_read_failure_raises(self):
        """Should raise RuntimeError when sandbox read fails."""
        sandbox = Mock()
        read_res = Mock()
        read_res.status = SandboxStatus.ERROR
        read_res.error = "permission denied"
        sandbox.read_file.return_value = read_res
        with pytest.raises(RuntimeError, match="permission denied"):
            _read_text_lossy("/f", sandbox)

    def test_bytearray_content(self):
        """bytearray (not just bytes) should also work."""
        raw = bytearray("你好\n".encode("utf-8"))
        sandbox = Mock()
        read_res = Mock()
        read_res.status = SandboxStatus.SUCCESS
        read_res.content = raw
        sandbox.read_file.return_value = read_res
        assert _read_text_lossy("/f", sandbox) == "你好\n"


# ---------------------------------------------------------------------------
# End-to-end read_file encoding tests (through full read_file function)
# ---------------------------------------------------------------------------


class TestReadFileEncoding:
    """End-to-end tests for read_file with various encodings."""

    def test_utf8_file_reads_correctly(self):
        raw = "第一行\n第二行\n第三行\n".encode("utf-8")
        result = _call_read_file_with_bytes(raw)
        assert "error" not in result
        assert "第一行" in result["content"]
        assert "第二行" in result["content"]
        assert "第三行" in result["content"]

    def test_ascii_file_reads_correctly(self):
        raw = b"line one\nline two\nline three\n"
        result = _call_read_file_with_bytes(raw)
        assert "error" not in result
        assert "line one" in result["content"]

    def test_large_ascii_prefix_then_chinese(self):
        """The exact scenario that broke chardet: >10KB ASCII then Chinese."""
        ascii_prefix = ("x" * 100 + "\n") * 150  # ~15KB of ASCII
        chinese_suffix = "中文内容在这里\n最后一行\n"
        raw = (ascii_prefix + chinese_suffix).encode("utf-8")
        result = _call_read_file_with_bytes(raw)
        assert "error" not in result
        assert "中文内容在这里" in result["content"]

    def test_non_utf8_file_does_not_error(self):
        """GBK file should not crash — invalid bytes replaced with U+FFFD."""
        raw = "这是GBK编码".encode("gbk")
        result = _call_read_file_with_bytes(raw)
        assert "error" not in result
        assert "\ufffd" in result["content"]

    def test_latin1_file_does_not_error(self):
        """Latin-1 file should not crash."""
        raw = "café résumé naïve\n".encode("latin-1")
        result = _call_read_file_with_bytes(raw)
        assert "error" not in result
        assert "caf" in result["content"]

    def test_binary_garbage_does_not_crash(self):
        """Random bytes should produce replacement chars, not an exception."""
        import os

        raw = os.urandom(256)
        result = _call_read_file_with_bytes(raw)
        assert "error" not in result

    def test_utf8_bom_file(self):
        """UTF-8 BOM should pass through without issues."""
        raw = b"\xef\xbb\xbf" + "带BOM的文件\n".encode("utf-8")
        result = _call_read_file_with_bytes(raw)
        assert "error" not in result
        assert "带BOM的文件" in result["content"]

    def test_crlf_line_endings(self):
        """Windows-style CRLF line endings should be handled."""
        raw = b"line1\r\nline2\r\nline3\r\n"
        result = _call_read_file_with_bytes(raw)
        assert "error" not in result
        assert "line1" in result["content"]
        assert "line2" in result["content"]

    def test_single_read_call_no_double_io(self):
        """Verify sandbox.read_file is called exactly once (binary=True), no double I/O."""
        raw = b"test content\n"
        sandbox = _make_sandbox_with_bytes(raw)
        agent_state = _make_agent_state(sandbox)
        read_file(file_path="test.txt", agent_state=agent_state)
        # read_file should only be called once with binary=True
        sandbox.read_file.assert_called_once()
        call_kwargs = sandbox.read_file.call_args
        assert call_kwargs[1].get("binary") is True or (len(call_kwargs[0]) > 2 and call_kwargs[0][2] is True)


# ---------------------------------------------------------------------------
# End-to-end tests with real temporary files (LocalSandbox)
# ---------------------------------------------------------------------------


def _write_tmp_file(tmp_path: Path, name: str, raw: bytes) -> str:
    """Write raw bytes to a file under tmp_path and return the filename."""
    p = tmp_path / name
    p.write_bytes(raw)
    return name


def _read_file_e2e(
    tmp_path: Path,
    filename: str,
    offset: int | None = None,
    limit: int | None = None,
) -> dict:
    """Create a real LocalSandbox pointing at tmp_path and call read_file."""
    from nexau.archs.sandbox.local_sandbox import LocalSandbox

    sandbox = LocalSandbox(work_dir=tmp_path)
    agent_state = Mock()
    agent_state.get_sandbox.return_value = sandbox
    return read_file(file_path=filename, agent_state=agent_state, offset=offset, limit=limit)


class TestReadFileWithTempFiles:
    """End-to-end tests using real temporary files on disk via LocalSandbox."""

    # -- UTF-8 variants --

    def test_utf8_ascii_file(self, tmp_path: Path):
        name = _write_tmp_file(tmp_path, "ascii.txt", b"hello world\nline two\n")
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "hello world" in result["content"]
        assert "line two" in result["content"]

    def test_utf8_chinese_file(self, tmp_path: Path):
        raw = "第一行\n第二行\n第三行\n".encode("utf-8")
        name = _write_tmp_file(tmp_path, "chinese.txt", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "第一行" in result["content"]
        assert "第三行" in result["content"]

    def test_utf8_japanese_file(self, tmp_path: Path):
        raw = "こんにちは世界\n二行目です\n".encode("utf-8")
        name = _write_tmp_file(tmp_path, "jp.txt", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "こんにちは世界" in result["content"]

    def test_utf8_korean_file(self, tmp_path: Path):
        raw = "한국어 테스트\n두 번째 줄\n".encode("utf-8")
        name = _write_tmp_file(tmp_path, "kr.txt", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "한국어 테스트" in result["content"]

    def test_utf8_emoji_file(self, tmp_path: Path):
        raw = "rocket 🚀\nearth 🌍\nparty 🎉\n".encode("utf-8")
        name = _write_tmp_file(tmp_path, "emoji.txt", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "🚀" in result["content"]
        assert "🌍" in result["content"]

    def test_utf8_mixed_scripts_file(self, tmp_path: Path):
        raw = "English 中文 日本語 한국어 العربية\n".encode("utf-8")
        name = _write_tmp_file(tmp_path, "mixed.txt", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "中文" in result["content"]
        assert "العربية" in result["content"]

    def test_utf8_bom_file(self, tmp_path: Path):
        raw = b"\xef\xbb\xbf" + "带BOM的UTF-8文件\n第二行\n".encode("utf-8")
        name = _write_tmp_file(tmp_path, "bom.txt", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "带BOM的UTF-8文件" in result["content"]

    # -- non-UTF-8 encodings: lossy decode --

    def test_gbk_file_lossy(self, tmp_path: Path):
        """GBK-encoded file — invalid UTF-8 bytes become U+FFFD."""
        raw = "你好世界，这是GBK编码".encode("gbk")
        name = _write_tmp_file(tmp_path, "gbk.txt", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "\ufffd" in result["content"]

    def test_latin1_file_lossy(self, tmp_path: Path):
        """Latin-1 encoded file — non-ASCII bytes become U+FFFD."""
        raw = "café résumé naïve über\n".encode("latin-1")
        name = _write_tmp_file(tmp_path, "latin1.txt", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "caf" in result["content"]
        assert "\ufffd" in result["content"]

    def test_shift_jis_file_lossy(self, tmp_path: Path):
        """Shift-JIS encoded file — invalid bytes become U+FFFD."""
        raw = "日本語テスト".encode("shift_jis")
        name = _write_tmp_file(tmp_path, "sjis.txt", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "\ufffd" in result["content"]

    def test_euc_kr_file_lossy(self, tmp_path: Path):
        """EUC-KR encoded file — invalid bytes become U+FFFD."""
        raw = "한국어 테스트".encode("euc_kr")
        name = _write_tmp_file(tmp_path, "euckr.txt", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "\ufffd" in result["content"]

    # -- the chardet bug scenario --

    def test_large_ascii_prefix_then_chinese(self, tmp_path: Path):
        """Exact scenario that broke chardet: >10KB ASCII followed by Chinese."""
        ascii_prefix = ("x" * 100 + "\n") * 150  # ~15KB of ASCII
        chinese_suffix = "中文内容在这里\n最后一行\n"
        raw = (ascii_prefix + chinese_suffix).encode("utf-8")
        name = _write_tmp_file(tmp_path, "big_ascii_then_cn.txt", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "中文内容在这里" in result["content"]

    # -- edge cases --

    def test_empty_file(self, tmp_path: Path):
        name = _write_tmp_file(tmp_path, "empty.txt", b"")
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "Read 0 lines" in result["returnDisplay"]

    def test_single_newline_file(self, tmp_path: Path):
        name = _write_tmp_file(tmp_path, "newline.txt", b"\n")
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result

    def test_crlf_line_endings(self, tmp_path: Path):
        raw = b"line1\r\nline2\r\nline3\r\n"
        name = _write_tmp_file(tmp_path, "crlf.txt", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "line1" in result["content"]
        assert "line2" in result["content"]
        assert "line3" in result["content"]

    def test_mixed_line_endings(self, tmp_path: Path):
        raw = b"unix\nwindows\r\nold_mac\rend"
        name = _write_tmp_file(tmp_path, "mixed_eol.txt", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "unix" in result["content"]
        assert "windows" in result["content"]

    def test_binary_garbage_file(self, tmp_path: Path):
        """Random binary bytes should produce replacement chars, not crash."""
        import os

        raw = os.urandom(512)
        name = _write_tmp_file(tmp_path, "garbage.bin", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result

    def test_long_line_truncation(self, tmp_path: Path):
        """Lines > 2000 chars should be truncated."""
        long_line = "x" * 3000
        raw = f"short\n{long_line}\nend\n".encode("utf-8")
        name = _write_tmp_file(tmp_path, "longline.txt", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "... [truncated]" in result["content"]
        assert "x" * 3000 not in result["content"]

    # -- offset / limit --

    def test_offset_and_limit(self, tmp_path: Path):
        lines = "\n".join(f"line {i}" for i in range(100)) + "\n"
        raw = lines.encode("utf-8")
        name = _write_tmp_file(tmp_path, "many_lines.txt", raw)
        result = _read_file_e2e(tmp_path, name, offset=10, limit=5)
        assert "error" not in result
        assert "line 10" in result["content"]
        assert "line 14" in result["content"]
        assert "line 9" not in result["content"]
        assert "line 15" not in result["content"]

    def test_offset_beyond_eof(self, tmp_path: Path):
        raw = b"one\ntwo\nthree\n"
        name = _write_tmp_file(tmp_path, "short.txt", raw)
        result = _read_file_e2e(tmp_path, name, offset=999)
        assert "error" not in result
        assert "Read 3 lines" in result["returnDisplay"] or result["content"].strip() == ""

    # -- file not found / directory --

    def test_file_not_found(self, tmp_path: Path):
        result = _read_file_e2e(tmp_path, "does_not_exist.txt")
        assert result.get("error") is not None
        assert "not found" in result["error"]["message"].lower() or "not exist" in result["error"]["message"].lower()

    def test_directory_returns_error(self, tmp_path: Path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        result = _read_file_e2e(tmp_path, "subdir")
        assert result.get("error") is not None
        assert result["error"]["type"] == "PATH_IS_DIRECTORY"

    # -- real source code files --

    def test_python_source_file(self, tmp_path: Path):
        code = (
            "# -*- coding: utf-8 -*-\n"
            "def greet(name: str) -> str:\n"
            '    """问候函数。"""\n'
            "    return f'你好, {name}!'\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    print(greet('世界'))\n"
        )
        raw = code.encode("utf-8")
        name = _write_tmp_file(tmp_path, "hello.py", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "def greet" in result["content"]
        assert "问候函数" in result["content"]

    def test_json_file(self, tmp_path: Path):
        raw = '{"name": "测试", "values": [1, 2, 3], "emoji": "🎯"}\n'.encode("utf-8")
        name = _write_tmp_file(tmp_path, "data.json", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "测试" in result["content"]
        assert "🎯" in result["content"]

    def test_markdown_file_with_chinese(self, tmp_path: Path):
        raw = "# 标题\n\n这是一段**中文** Markdown。\n\n- 列表项 1\n- 列表项 2\n".encode("utf-8")
        name = _write_tmp_file(tmp_path, "readme.md", raw)
        result = _read_file_e2e(tmp_path, name)
        assert "error" not in result
        assert "标题" in result["content"]
        assert "列表项" in result["content"]


class TestHeadTruncateLines:
    """Unit tests for _head_truncate_lines helper."""

    def test_no_truncation_when_under_budget(self):
        """Lines that fit within budget should pass through unchanged."""
        lines = ["short"] * 10
        result, omitted, omitted_chars = _head_truncate_lines(lines)
        assert result == lines
        assert omitted == 0
        assert omitted_chars == 0

    def test_truncation_drops_trailing_lines(self):
        """When over budget, trailing lines should be dropped."""
        # 100 lines × 500 chars each = ~50K chars, over 30K budget
        lines = ["x" * 500] * 100
        result, omitted, omitted_chars = _head_truncate_lines(lines)
        assert omitted > 0
        assert len(result) == 100 - omitted
        # Kept lines are from the beginning
        assert result == lines[: len(result)]

    def test_no_truncation_when_exactly_at_budget(self):
        """Lines totaling exactly the budget should not be truncated."""
        line_len = 299
        n = 100
        total = n * line_len + (n - 1)
        assert total <= MAX_TOTAL_OUTPUT_CHARS
        lines = ["b" * line_len] * n
        result, omitted, _ = _head_truncate_lines(lines)
        assert omitted == 0
        assert result == lines

    def test_single_very_long_line_kept(self):
        """A single line (even if very long) is always kept (at least 1 line guard)."""
        lines = ["z" * 100_000]
        result, omitted, _ = _head_truncate_lines(lines)
        assert result == lines
        assert omitted == 0

    def test_empty_input(self):
        """Empty line list should pass through."""
        result, omitted, _ = _head_truncate_lines([])
        assert result == []
        assert omitted == 0

    def test_custom_budget(self):
        """Custom max_chars should be respected."""
        lines = ["hello world"] * 50  # 50 × 11 + 49 = 599 chars
        result, omitted, _ = _head_truncate_lines(lines, max_chars=100)
        assert omitted > 0
        assert len(result) < 50


class TestReadFileHeadTruncateIntegration:
    """Integration tests: read_file with total output budget truncation."""

    def _make_sandbox(self, content: str):
        """Create a mock sandbox that returns the given text content."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        raw_bytes = content.encode("utf-8")

        info = Mock()
        info.is_directory = False
        info.size = len(raw_bytes)
        sandbox.get_file_info.return_value = info

        read_res = Mock()
        read_res.status = SandboxStatus.SUCCESS
        read_res.content = raw_bytes

        sandbox.read_file.return_value = read_res
        return sandbox

    def test_small_file_no_truncation(self):
        """Files under the output budget should not trigger truncation."""
        content = "\n".join(f"line {i}" for i in range(100))
        sandbox = self._make_sandbox(content)
        result = read_file(file_path="small.txt", agent_state=_make_agent_state(sandbox))

        assert "error" not in result or result.get("error") is None
        assert "output truncated" not in result["returnDisplay"]
        assert "100| " in result["content"] or "Read 100 lines" in result["returnDisplay"]

    def test_large_file_triggers_truncation(self):
        """Files exceeding the output budget should have trailing lines truncated."""
        # 500 lines × 200 chars each ≈ 100K chars, well over 30K budget
        content = "\n".join("y" * 200 for _ in range(500))
        sandbox = self._make_sandbox(content)
        result = read_file(file_path="big.txt", agent_state=_make_agent_state(sandbox))

        assert "error" not in result or result.get("error") is None
        llm_content = result["content"]
        display = result["returnDisplay"]

        # Should indicate output was truncated
        assert "output truncated" in display
        assert "chars truncated" in llm_content
        assert "use offset/limit to read further" in llm_content

        # First line should be present, last line should NOT (only head kept)
        assert "1| " in llm_content
        assert "500| " not in llm_content

    def test_truncation_with_offset(self):
        """Truncation should work correctly with offset parameter."""
        content = "\n".join(f"{'z' * 199}{i:04d}" for i in range(600))
        sandbox = self._make_sandbox(content)
        result = read_file(
            file_path="offset.txt",
            offset=100,
            limit=400,
            agent_state=_make_agent_state(sandbox),
        )

        llm_content = result["content"]
        assert "101| " in llm_content

    def test_combined_per_line_and_head_truncation(self):
        """Both per-line and head truncation can trigger together."""
        # 300 lines, each 3000 chars → per-line truncates to 2000, total still ~600K
        content = "\n".join("w" * 3000 for _ in range(300))
        sandbox = self._make_sandbox(content)
        result = read_file(file_path="combo.txt", agent_state=_make_agent_state(sandbox))

        display = result["returnDisplay"]
        assert "output truncated" in display
        assert "some lines were shortened" in display
