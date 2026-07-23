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

"""Persistence-entry image resize: HistoryList downscales oversized images
(top-level ImageBlock, ToolResultBlock.content, and untyped raw_output) to the
token-budget pixel cap, in place, before they reach persistence — so stored
history and the next LLM turn both carry the smaller image."""

import asyncio
import base64
import io
import os

import pytest
from PIL import Image

from nexau.archs.main_sub.history_list import HistoryList
from nexau.archs.main_sub.utils.image_probe import (
    _MAX_DECODE_PIXELS,
    DEFAULT_IMAGE_MAX_PIXELS,
    MAX_IMAGE_BASE64_BYTES,
    OVERSIZED_IMAGE_PLACEHOLDER,
    image_exceeds_hard_limit,
    probe_dimensions,
    resize_base64_image_if_oversized,
)
from nexau.archs.session import AgentRunActionKey, AgentRunActionModel, SessionManager
from nexau.archs.session.orm import InMemoryDatabaseEngine
from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolResultBlock

# Comfortably over the ~3.75 MP area cap so the downscale is unambiguous.
_OVERSIZED_WH = (2400, 1800)
# Well under the cap → must be a no-op.
_WITHIN_BOUND_WH = (800, 600)


def _png_b64(width: int, height: int, *, noise: bool = False) -> str:
    """Base64 (no data: prefix) of a PNG. ``noise=True`` fills random pixels so
    the source PNG is large and downscaled JPEG is clearly smaller in bytes;
    solid color (default) is cheap and used where only dims/mime are asserted."""
    if noise:
        data = os.urandom(width * height * 3)
    else:
        data = bytes([70, 130, 180]) * (width * height)
    img = Image.frombytes("RGB", (width, height), data)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _decoded_pixels(b64: str) -> int:
    dims = probe_dimensions(base64.b64decode(b64))
    assert dims is not None, "resized output must be probeable"
    return dims[0] * dims[1]


def _frame_count(b64: str) -> int:
    """Count frames via the base-class ``seek``/``EOFError`` contract — avoids
    the subclass-only ``n_frames`` attribute so the test stays pyright-clean."""
    count = 0
    with Image.open(io.BytesIO(base64.b64decode(b64))) as img:
        try:
            while True:
                img.seek(count)
                count += 1
        except EOFError:
            pass
    return count


def _animated_gif_b64(width: int, height: int, frames: int) -> str:
    """A multi-frame (animated) GIF. Palette ("P") solid frames keep it small."""
    imgs = [Image.new("P", (width, height), color=idx * 40 % 256) for idx in range(frames)]
    buffer = io.BytesIO()
    imgs[0].save(buffer, format="GIF", save_all=True, append_images=imgs[1:], duration=100, loop=0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _large_solid_png_b64(width: int, height: int) -> str:
    """A decodable but very large solid-color PNG. Grayscale ("L") keeps the
    source buffer ~1 byte/pixel and the solid fill compresses to a few KB. Kept
    under Pillow's ~89.5 MP DecompressionBomb limit so the >60 MP client guard —
    not Pillow's own bomb check — is what skips it."""
    img = Image.new("L", (width, height), 128)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _oversized_b64_str() -> str:
    """A base64 string longer than the 20 MiB hard limit. Deliberately NOT a real
    image: ``image_exceeds_hard_limit``'s length gate fires before any decode, so
    the bytes never need to parse. Fast + deterministic vs. encoding a >20 MB PNG."""
    return "/" * (MAX_IMAGE_BASE64_BYTES + 4)


# ---------------------------------------------------------------------------
# ① Oversized top-level ImageBlock is downscaled on append
# ---------------------------------------------------------------------------
def test_oversized_top_level_image_downscaled_on_append():
    source_b64 = _png_b64(*_OVERSIZED_WH, noise=True)
    source_bytes = len(base64.b64decode(source_b64))
    block = ImageBlock(base64=source_b64, mime_type="image/png")

    history = HistoryList()
    history.append(Message(role=Role.USER, content=[block]))

    # In-place mutation: same object the LLM sees next turn.
    assert block.mime_type == "image/jpeg"
    assert _decoded_pixels(block.base64) <= DEFAULT_IMAGE_MAX_PIXELS
    assert len(base64.b64decode(block.base64)) < source_bytes


# ---------------------------------------------------------------------------
# ② Within-bound image is a no-op (idempotent — restored/relayed small images)
# ---------------------------------------------------------------------------
def test_within_bound_image_untouched_on_append():
    source_b64 = _png_b64(*_WITHIN_BOUND_WH)
    block = ImageBlock(base64=source_b64, mime_type="image/png")

    history = HistoryList()
    history.append(Message(role=Role.USER, content=[block]))

    assert block.base64 == source_b64
    assert block.mime_type == "image/png"


# ---------------------------------------------------------------------------
# ③ url-only image (base64 is None) is skipped without error
# ---------------------------------------------------------------------------
def test_url_only_image_skipped():
    block = ImageBlock(url="https://example.com/photo.png")

    history = HistoryList()
    history.append(Message(role=Role.USER, content=[block]))  # must not raise

    assert block.url == "https://example.com/photo.png"
    assert block.base64 is None


# ---------------------------------------------------------------------------
# ④ ImageBlock nested in ToolResultBlock.content is downscaled
# ---------------------------------------------------------------------------
def test_raw_output_url_only_image_untouched():
    """A plain http image_url (no base64 payload) in raw_output is left as-is."""
    raw_output = {"content": {"type": "image", "image_url": "https://example.com/a.png"}}
    tool_result = ToolResultBlock(tool_use_id="call_1", content="ok", raw_output=raw_output)

    HistoryList().append(Message(role=Role.TOOL, content=[tool_result]))

    assert raw_output["content"]["image_url"] == "https://example.com/a.png"


# ---------------------------------------------------------------------------
# extend() and replace_all() go through the same resize hook
# ---------------------------------------------------------------------------
def test_extend_downscales_oversized_images():
    block = ImageBlock(base64=_png_b64(*_OVERSIZED_WH), mime_type="image/png")

    HistoryList().extend([Message(role=Role.USER, content=[block])])

    assert block.mime_type == "image/jpeg"
    assert _decoded_pixels(block.base64) <= DEFAULT_IMAGE_MAX_PIXELS


def test_replace_all_downscales_oversized_images():
    block = ImageBlock(base64=_png_b64(*_OVERSIZED_WH), mime_type="image/png")

    HistoryList().replace_all([Message(role=Role.USER, content=[block])])

    assert block.mime_type == "image/jpeg"
    assert _decoded_pixels(block.base64) <= DEFAULT_IMAGE_MAX_PIXELS


# ---------------------------------------------------------------------------
# Idempotency + graceful degradation
# ---------------------------------------------------------------------------
def test_resize_is_idempotent():
    """Re-appending an already-resized message must not shrink it again."""
    block = ImageBlock(base64=_png_b64(*_OVERSIZED_WH), mime_type="image/png")
    message = Message(role=Role.USER, content=[block])

    history = HistoryList()
    history.append(message)
    after_first = block.base64

    history.append(message)  # second pass: image is now within bound → no-op
    assert block.base64 == after_first
    assert block.mime_type == "image/jpeg"


def test_resize_function_second_call_is_noop():
    resized = resize_base64_image_if_oversized(_png_b64(*_OVERSIZED_WH), "image/png")
    assert resized is not None
    new_b64, _ = resized
    assert resize_base64_image_if_oversized(new_b64, "image/jpeg") is None


def test_undecodable_image_does_not_fail_history_write():
    """A base64 that isn't a real image must be left untouched, never raise."""
    junk_b64 = base64.b64encode(b"this is not an image at all").decode("utf-8")
    block = ImageBlock(base64=junk_b64, mime_type="image/png")

    history = HistoryList()
    history.append(Message(role=Role.USER, content=[block]))  # must not raise

    assert block.base64 == junk_b64


# ---------------------------------------------------------------------------
# extend() a mixed batch: oversized downscaled, within-bound + text untouched
# ---------------------------------------------------------------------------
def test_extend_mixed_batch_processed_per_message():
    oversized = ImageBlock(base64=_png_b64(*_OVERSIZED_WH, noise=True), mime_type="image/png")
    within = ImageBlock(base64=_png_b64(*_WITHIN_BOUND_WH), mime_type="image/png")
    within_src = within.base64
    text_block = TextBlock(text="plain text, no image")

    HistoryList().extend(
        [
            Message(role=Role.USER, content=[oversized]),
            Message(role=Role.USER, content=[text_block]),
            Message(role=Role.USER, content=[within]),
        ]
    )

    # oversized → downscaled to JPEG within budget
    assert oversized.mime_type == "image/jpeg"
    assert _decoded_pixels(oversized.base64) <= DEFAULT_IMAGE_MAX_PIXELS
    # within-bound → untouched
    assert within.mime_type == "image/png"
    assert within.base64 == within_src
    # plain text → untouched
    assert text_block.text == "plain text, no image"


# ---------------------------------------------------------------------------
# Deep-nested raw_output (dict→list→dict) and a list-rooted raw_output
# ---------------------------------------------------------------------------
def test_raw_output_non_image_base64_untouched():
    pdf_b64 = base64.b64encode(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n" + os.urandom(4096)).decode("utf-8")
    entry = {"base64": pdf_b64, "media_type": "application/pdf"}
    raw_output = {"files": [entry]}
    tool_result = ToolResultBlock(tool_use_id="c1", content="ok", raw_output=raw_output)

    HistoryList().append(Message(role=Role.TOOL, content=[tool_result]))  # must not raise

    assert entry["base64"] == pdf_b64
    assert entry["media_type"] == "application/pdf"


# ---------------------------------------------------------------------------
# Animated image is skipped (kept as-is), never flattened to a single frame
# ---------------------------------------------------------------------------
def test_animated_gif_skipped_not_flattened():
    src_b64 = _animated_gif_b64(*_OVERSIZED_WH, frames=3)
    # Sanity: oversized (a *static* image this big would downscale to JPEG) and
    # genuinely multi-frame.
    dims = probe_dimensions(base64.b64decode(src_b64))
    assert dims is not None and dims[0] * dims[1] > DEFAULT_IMAGE_MAX_PIXELS
    assert _frame_count(src_b64) == 3

    block = ImageBlock(base64=src_b64, mime_type="image/gif")
    HistoryList().append(Message(role=Role.USER, content=[block]))

    # Animated → skipped: original bytes + frame count preserved (not flattened).
    assert block.base64 == src_b64
    assert block.mime_type == "image/gif"
    assert _frame_count(block.base64) == 3


# ---------------------------------------------------------------------------
# Oversized-beyond-guard image (>60 MP) is OMITTED — the pixel hard gate fires
# before any full decode, so the block is replaced with a placeholder TextBlock
# rather than resized or kept.
# ---------------------------------------------------------------------------
def test_oversized_beyond_decode_guard_omitted():
    # 8000x7700 = 61.6 MP: over the 60 MP decode guard, under Pillow's ~89.5 MP
    # bomb limit — so the *pixel* hard gate fires. Solid PNG bytes stay tiny, so
    # it's the pixel gate (not the base64-length gate) that omits it.
    src_b64 = _large_solid_png_b64(8000, 7700)
    dims = probe_dimensions(base64.b64decode(src_b64))
    assert dims is not None and dims[0] * dims[1] > _MAX_DECODE_PIXELS
    assert len(src_b64) <= MAX_IMAGE_BASE64_BYTES  # the length gate must NOT be what fires

    block = ImageBlock(base64=src_b64, mime_type="image/png")
    message = Message(role=Role.USER, content=[block])
    HistoryList().append(message)  # must not raise (no full decode)

    replaced = message.content[0]
    assert isinstance(replaced, TextBlock)
    assert replaced.text == OVERSIZED_IMAGE_PLACEHOLDER
    assert block not in message.content


# ---------------------------------------------------------------------------
# Hard size gate → omit placeholder (base64 > 20 MiB), before any resize/decode
# ---------------------------------------------------------------------------
def test_image_exceeds_hard_limit_gates():
    # base64-length gate: fires on a >20 MiB string without decoding it.
    assert image_exceeds_hard_limit(_oversized_b64_str()) is True
    # pixel gate: tiny bytes but >60 MP dimensions.
    assert image_exceeds_hard_limit(_large_solid_png_b64(8000, 7700)) is True
    # within both gates → False (a normal oversized image still *resizes*, not omit).
    assert image_exceeds_hard_limit(_png_b64(*_OVERSIZED_WH)) is False
    assert image_exceeds_hard_limit(_png_b64(*_WITHIN_BOUND_WH)) is False


def test_oversized_base64_top_level_image_omitted():
    """A >20 MiB base64 ImageBlock is replaced by a placeholder TextBlock; the
    original image is dropped from content and never decoded."""
    block = ImageBlock(base64=_oversized_b64_str(), mime_type="image/png")
    message = Message(role=Role.USER, content=[block])

    HistoryList().append(message)

    replaced = message.content[0]
    assert isinstance(replaced, TextBlock)
    assert replaced.text == OVERSIZED_IMAGE_PLACEHOLDER
    assert block not in message.content


@pytest.fixture
def engine():
    return InMemoryDatabaseEngine()


@pytest.fixture
def session_manager(engine):
    return SessionManager(engine=engine)


@pytest.fixture
def history_key():
    return AgentRunActionKey(user_id="u1", session_id="s1", agent_id="a1")


def test_persisted_history_stores_downscaled_image(session_manager, history_key, engine):
    async def run():
        await engine.setup_models([AgentRunActionModel])

        history = HistoryList(
            session_manager=session_manager,
            history_key=history_key,
            run_id="run_001",
            root_run_id="run_001",
            agent_name="a1",
        )
        source_b64 = _png_b64(*_OVERSIZED_WH, noise=True)
        history.append(Message(role=Role.USER, content=[ImageBlock(base64=source_b64, mime_type="image/png")]))
        history.flush()
        await asyncio.sleep(0.01)

        loaded = await session_manager.agent_run_action.load_messages(key=history_key)
        assert len(loaded) == 1
        stored = loaded[0].content[0]
        assert isinstance(stored, ImageBlock)
        assert stored.mime_type == "image/jpeg"
        assert stored.base64 is not None
        assert _decoded_pixels(stored.base64) <= DEFAULT_IMAGE_MAX_PIXELS
        assert len(base64.b64decode(stored.base64)) < len(base64.b64decode(source_b64))

    asyncio.run(run())


class TestReviewFixGeometryAndProber:
    """#601 review 修复的验收:极端长宽比封顶、prober 扩展、bomb 兜底、非图保护。"""

    def test_extreme_aspect_ratio_is_capped_by_long_edge(self) -> None:
        from nexau.archs.main_sub.utils.image_probe import MAX_TARGET_LONG_EDGE, area_capped_dimensions

        # 4,000,000x1:面积仅 4MP(界内)但 patch 计费 ~14 万 token。
        target = area_capped_dimensions(4_000_000, 1, DEFAULT_IMAGE_MAX_PIXELS)
        assert target is not None
        target_w, target_h = target
        assert max(target_w, target_h) <= MAX_TARGET_LONG_EDGE
        assert target_w * target_h <= DEFAULT_IMAGE_MAX_PIXELS

    def test_long_edge_over_limit_triggers_resize_even_when_area_is_within(self) -> None:
        from nexau.archs.main_sub.utils.image_probe import area_capped_dimensions

        # 20000x100 = 2MP(面积界内)但长边超 8000 → 必须给出目标。
        target = area_capped_dimensions(20_000, 100, DEFAULT_IMAGE_MAX_PIXELS)
        assert target is not None
        assert max(target) <= 8000

    def test_normal_budget_sized_image_is_untouched_by_long_edge_cap(self) -> None:
        from nexau.archs.main_sub.utils.image_probe import area_capped_dimensions

        # 官方高分档示例尺寸:面积贴 cap、长边远小于 8000 → 仍是 no-op。
        assert area_capped_dimensions(2582, 1452, DEFAULT_IMAGE_MAX_PIXELS) is None

    def test_webp_prober_parses_vp8x_canvas(self) -> None:
        buf = bytearray(30)
        buf[0:4] = b"RIFF"
        buf[8:12] = b"WEBP"
        buf[12:16] = b"VP8X"
        width, height = 4000 - 1, 3000 - 1
        buf[24:27] = width.to_bytes(3, "little")
        buf[27:30] = height.to_bytes(3, "little")
        assert probe_dimensions(bytes(buf)) == (4000, 3000)

    def test_bmp_core_header_parses_u16_dimensions(self) -> None:
        import struct

        # OS/2 BITMAPCOREHEADER(DIB size=12):width/height 是 u16@18/20。
        buf = bytearray(26)
        buf[0:2] = b"BM"
        buf[14:18] = struct.pack("<I", 12)
        buf[18:20] = struct.pack("<H", 100)
        buf[20:22] = struct.pack("<H", 100)
        assert probe_dimensions(bytes(buf)) == (100, 100)

    def test_pillow_header_fallback_covers_unprobeable_formats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nexau.archs.main_sub.utils import image_probe

        # TIFF 不在手写 prober 覆盖内 → Pillow lazy open 兜底探出尺寸;
        # 把 60MP 护栏调小后,hard gate 必须借 fallback 拦下它。
        img = Image.new("RGB", (64, 48), (10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="TIFF")
        b64 = base64.b64encode(buf.getvalue()).decode()
        assert probe_dimensions(buf.getvalue()) is None
        monkeypatch.setattr(image_probe, "_MAX_DECODE_PIXELS", 1_000)
        assert image_exceeds_hard_limit(b64) is True

    def test_resize_refuses_to_decode_over_guard_even_if_gate_missed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nexau.archs.main_sub.utils import image_probe

        # 直接调用 resize(绕过 hard gate)时,open 后的守卫拒绝全解码,原样保留。
        b64 = _png_b64(400, 300)
        monkeypatch.setattr(image_probe, "_MAX_DECODE_PIXELS", 1_000)
        monkeypatch.setattr(image_probe, "DEFAULT_IMAGE_MAX_PIXELS", 100)
        assert resize_base64_image_if_oversized(b64, "image/png") is None


class TestInboundImageEntryGate:
    """#601 入口硬闸:用户直传超限图 fail-fast,不静默进 history。"""

    def test_oversized_user_image_is_rejected_with_actionable_error(self) -> None:
        from nexau.archs.main_sub.utils.image_probe import (
            OversizedInboundImageError,
            ensure_inbound_images_within_limits,
        )

        oversized = Message(
            role=Role.USER,
            content=[ImageBlock(base64="A" * (MAX_IMAGE_BASE64_BYTES + 8))],
        )
        with pytest.raises(OversizedInboundImageError) as excinfo:
            ensure_inbound_images_within_limits([oversized])
        assert "Compress or downscale" in str(excinfo.value)
        assert "message #1" in str(excinfo.value)

    def test_oversized_pixels_probed_image_is_rejected(self) -> None:
        # 9000x9000 PNG header(81MP > 60MP)but tiny in bytes.
        import struct as _struct

        from nexau.archs.main_sub.utils.image_probe import (
            OversizedInboundImageError,
            ensure_inbound_images_within_limits,
        )

        header = b"\x89PNG\r\n\x1a\n" + _struct.pack(">I", 13) + b"IHDR" + _struct.pack(">II", 9000, 9000)
        message = Message(
            role=Role.USER,
            content=[ImageBlock(base64=base64.b64encode(header).decode())],
        )
        with pytest.raises(OversizedInboundImageError):
            ensure_inbound_images_within_limits([message])

    def test_within_bound_images_and_non_user_roles_pass(self) -> None:
        from nexau.archs.main_sub.utils.image_probe import ensure_inbound_images_within_limits

        small = Message(role=Role.USER, content=[ImageBlock(base64=_png_b64(*_WITHIN_BOUND_WH))])
        tool_msg = Message(
            role=Role.TOOL,
            content=[ImageBlock(base64="A" * (MAX_IMAGE_BASE64_BYTES + 8))],
        )
        url_only = Message(role=Role.USER, content=[ImageBlock(url="https://example.com/big.png")])
        ensure_inbound_images_within_limits([small, tool_msg, url_only])


class TestToolOutputIsLeftUntouched:
    """#601 设计边界:工具产出(ToolResult 嵌套图 / raw_output)框架不碰 ——
    限制与压缩由工具作者自己负责;builtin 读工具已在读路径压缩。"""

    def _run_through_history(self, message: Message) -> None:
        history = HistoryList()
        history.append(message)

    def test_nested_toolresult_image_is_not_resized_or_omitted(self) -> None:
        oversized_b64 = _png_b64(*_OVERSIZED_WH)
        block = ToolResultBlock(
            tool_use_id="t1",
            content=[ImageBlock(base64=oversized_b64, mime_type="image/png")],
        )
        message = Message(role=Role.TOOL, content=[block])
        self._run_through_history(message)
        inner = message.content[0]
        assert isinstance(inner, ToolResultBlock)
        assert isinstance(inner.content, list)
        image = inner.content[0]
        assert isinstance(image, ImageBlock)
        assert image.base64 == oversized_b64
        assert image.mime_type == "image/png"

    def test_raw_output_payloads_are_not_touched(self) -> None:
        huge = "A" * (MAX_IMAGE_BASE64_BYTES + 8)
        data_uri = "data:image/png;base64," + huge
        block = ToolResultBlock(
            tool_use_id="t1",
            content="ok",
            raw_output={
                "files": [{"base64": huge, "media_type": "image/png"}],
                "preview": data_uri,
            },
        )
        message = Message(role=Role.TOOL, content=[block])
        self._run_through_history(message)
        inner = message.content[0]
        assert isinstance(inner, ToolResultBlock)
        raw = inner.raw_output
        assert isinstance(raw, dict)
        files = raw["files"]
        assert isinstance(files, list)
        first_file = files[0]
        assert isinstance(first_file, dict)
        assert first_file["base64"] == huge
        assert raw["preview"] == data_uri

    def test_top_level_user_image_is_still_processed(self) -> None:
        message = Message(
            role=Role.USER,
            content=[ImageBlock(base64="A" * (MAX_IMAGE_BASE64_BYTES + 8))],
        )
        self._run_through_history(message)
        replaced = message.content[0]
        assert isinstance(replaced, TextBlock)
        assert replaced.text == OVERSIZED_IMAGE_PLACEHOLDER
