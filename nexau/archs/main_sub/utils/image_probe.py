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

"""Image header probing + official token estimation, shared by the token
counter and the read-tool downscale step.

Incident fix (session bf6ef5c923ce; Rust counterpart nexau-rs#94): both the
token estimator (`token_counter._estimate_image_tokens`) and the read-tool
downscale step (`read_visual_file`) need an image's real pixel dimensions —
the former to charge context cost, the latter to decide the exact target size
that keeps one image within its token budget. Parsing the format header is
enough for both: no full pixel decode, no image-codec dependency.

Token cost uses Anthropic's official patch formula
(`estimate_tokens_from_dimensions`); it lives here so the counter and the
downscaler cannot drift apart inside this repo, and the same 28px patch size
must stay in sync with the Rust `nexau-rs` runtime by convention.
"""

import base64
import binascii
import logging
import math
import struct
from collections.abc import Sequence
from io import BytesIO
from typing import Final

from PIL import Image

logger = logging.getLogger(__name__)

# Anthropic's official vision token formula is patch-based: an image is tiled
# into 28x28-pixel patches and costs one token per patch —
# ceil(width/28) * ceil(height/28), i.e. 784 pixels per token (older docs'
# pixels/750 was an approximation; the patch formula matched northgate
# measurements to ±1 token across five sizes on 2026-07-07). We charge context
# cost at this official rate rather than a worst-channel calibration: images
# emitted by the read tools are already downscaled to a per-image token budget
# (`read_visual_file`, in official-formula tokens), so counting them by the
# same formula is exactly consistent. Keep the patch size in sync with the
# Rust `nexau-rs` runtime (token_counter.rs).
IMAGE_TOKEN_PATCH_SIZE: Final[int] = 28


def estimate_tokens_from_dimensions(width: int, height: int) -> int:
    """Official Anthropic vision token cost for a ``width x height`` image.

    One token per 28x28-pixel patch: ``ceil(w/28) * ceil(h/28)``. This is the
    server-side *pre-downscale* cost; a caller wanting the billed cost of an
    already-capped image passes the capped dimensions. A non-empty image never
    costs 0 (each axis contributes at least one patch).
    """
    if width <= 0 or height <= 0:
        return 0
    return math.ceil(width / IMAGE_TOKEN_PATCH_SIZE) * math.ceil(height / IMAGE_TOKEN_PATCH_SIZE)


# Every SOFn variant shares the `[precision:1][height:2 BE][width:2 BE]...`
# payload layout; DHT/DAC/RSTn/SOI/EOI are deliberately excluded.
_JPEG_SOF_MARKERS: Final[frozenset[int]] = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})


def probe_dimensions(data: bytes) -> tuple[int, int] | None:
    """Parse ``(width, height)`` from a (possibly truncated) image header.

    Tries each format nexau's own image tools emit (PNG, JPEG, GIF, BMP,
    WebP — the WebP prober mirrors nexau-rs so both runtimes classify the
    same bytes identically); an unrecognized or malformed header yields
    ``None``, which callers must treat as "size unknown — act
    conservatively", never as zero.
    """
    return (
        _probe_png_dimensions(data)
        or _probe_jpeg_dimensions(data)
        or _probe_gif_dimensions(data)
        or _probe_bmp_dimensions(data)
        or _probe_webp_dimensions(data)
    )


def _probe_webp_dimensions(data: bytes) -> tuple[int, int] | None:
    # RIFF container: "RIFF"[size:4]"WEBP" then a VP8 / VP8L / VP8X chunk.
    # Ported from nexau-rs `probe_webp_dimensions` for byte-for-byte parity.
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    chunk = data[12:16]
    if chunk == b"VP8X":
        # Extended: 24-bit LE canvas width/height minus one, at offset 24/27.
        width = 1 + (data[24] | (data[25] << 8) | (data[26] << 16))
        height = 1 + (data[27] | (data[28] << 8) | (data[29] << 16))
        return width, height
    if chunk == b"VP8L":
        # Lossless: signature byte 0x2f then 14-bit width-1 / height-1.
        if data[20] != 0x2F:
            return None
        bits = struct.unpack("<I", data[21:25])[0]
        return 1 + (bits & 0x3FFF), 1 + ((bits >> 14) & 0x3FFF)
    if chunk == b"VP8 ":
        # Lossy: frame tag (3B) + start code 9D 01 2A + 16-bit LE dims (14 bits used).
        if data[23] != 0x9D or data[24] != 0x01 or data[25] != 0x2A:
            return None
        width, height = struct.unpack("<HH", data[26:30])
        return int(width) & 0x3FFF, int(height) & 0x3FFF
    return None


def _probe_png_dimensions(data: bytes) -> tuple[int, int] | None:
    # 8-byte signature, then the IHDR chunk is always first:
    # [len:4][type:4="IHDR"][width:4 BE][height:4 BE]...
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def _probe_gif_dimensions(data: bytes) -> tuple[int, int] | None:
    # "GIF87a"|"GIF89a" then the logical screen descriptor: [width:2 LE][height:2 LE].
    if len(data) < 10 or data[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    width, height = struct.unpack("<HH", data[6:10])
    return int(width), int(height)


def _probe_bmp_dimensions(data: bytes) -> tuple[int, int] | None:
    # "BM" file header, then the DIB header. Its size field (offset 14, u32 LE)
    # discriminates the family: the legacy OS/2 BITMAPCOREHEADER (size 12)
    # stores width/height as u16 at offset 18/20, while the BITMAPINFOHEADER
    # family (V1/V4/V5, size >= 40) stores signed 32-bit LE at 18/22 (negative
    # height = top-down row order; magnitude is what we need). Parsing a CORE
    # header with the INFO layout used to fuse two u16 fields into a trillion-
    # pixel reading and mis-omit legitimate small images.
    if len(data) < 26 or data[:2] != b"BM":
        return None
    header_size = struct.unpack("<I", data[14:18])[0]
    if header_size == 12:
        width, height = struct.unpack("<HH", data[18:22])
        return int(width), int(height)
    width, height = struct.unpack("<ii", data[18:26])
    return abs(int(width)), abs(int(height))


def _probe_jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    # Scan segments from the SOI marker for a Start-Of-Frame marker and read its
    # height/width, skipping every other segment by its declared length. Bounded
    # by the caller-supplied bytes, so a truncated tail (SOF not yet reached)
    # safely yields None rather than reading out of bounds.
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    pos = 2
    length = len(data)
    while pos + 9 <= length:
        if data[pos] != 0xFF:
            return None  # not aligned on a marker boundary — bail rather than mis-scan
        marker = data[pos + 1]
        # Fill byte between markers: advance one and re-align.
        if marker == 0xFF:
            pos += 1
            continue
        # Standalone markers carry no length payload: TEM (0x01) and RSTn/SOI/EOI (0xD0-0xD9).
        if marker == 0x01 or 0xD0 <= marker <= 0xD9:
            pos += 2
            continue
        segment_len = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
        if marker in _JPEG_SOF_MARKERS:
            height, width = struct.unpack(">HH", data[pos + 5 : pos + 9])
            return int(width), int(height)
        pos += 2 + int(segment_len)
    return None


# ---------------------------------------------------------------------------
# Pixel-area budget + downscale geometry
#
# Incident fix (session bf6ef5c923ce; Rust counterpart nexau-rs#94): the same
# per-image token budget both the read tool (`read_visual_file`, via ffmpeg)
# and the persistence-time in-memory resize (`resize_base64_image_if_oversized`,
# via Pillow) cap images against. Kept here — the shared low-level image module,
# already the single source of the token-cost formula — so the read path, the
# persist path and the token counter can't drift to different exchange rates.
#
# The official Anthropic vision formula is patch-based: an image costs
# ceil(width/28) x ceil(height/28) visual tokens — one token per 28x28-pixel
# patch, i.e. 784 pixels per token. Per-image ceilings are tier-dependent;
# high-resolution models (Opus 4.7/4.8, Fable 5) allow ≤2576px / ≤4_784 tokens.
# Gateway channels implement those caps inconsistently, so we enforce the bound
# client-side. Budgets are denominated in official-formula tokens — the same
# formula `token_counter` charges context cost with — so a budgeted image and
# its accounted cost agree. Keep in sync with the Rust `nexau-rs` constants.
# ---------------------------------------------------------------------------
OFFICIAL_PIXELS_PER_TOKEN: Final[int] = 784

# Per-image token budget applied when no explicit cap is given. 4_784 matches
# the official high-resolution tier's own per-image ceiling — the tier the
# production model (claude-opus-4-8) is in: 4_784 x 784 ≈ 3.75 megapixels.
DEFAULT_IMAGE_TOKEN_BUDGET: Final[int] = 4_784

# Pixel-area ceiling derived from the default budget. Bounding area — not edge
# length — matches how token cost scales: an area cap prices every aspect ratio
# identically.
DEFAULT_IMAGE_MAX_PIXELS: Final[int] = DEFAULT_IMAGE_TOKEN_BUDGET * OFFICIAL_PIXELS_PER_TOKEN

# base64 前缀解码上限（界内快路径）：先只解码前缀喂 header prober 判界，界内图直接
# 返回、免掉整张大图的全解码。镜像 `token_counter._PROBE_PREFIX_BASE64_CHARS` ——
# 刻意保留为本模块常量而非 import，因为 `token_counter` 依赖本模块，反向 import
# 会形成循环依赖。PNG 尺寸在前 24 字节、JPEG 的 SOF 几乎总在数十 KB 内、GIF/BMP
# 在头几字节，200_000 base64 字符（约 150KB 解码）对四种格式都绰绰有余。
_PROBE_PREFIX_BASE64_CHARS: Final[int] = 200_000

# 全解码护栏：header 探出的像素面积超过此值时判定为超限图。远超正常图（默认封顶
# 约 3.75MP），又低于 Pillow DecompressionBomb 的默认阈值（`Image.MAX_IMAGE_PIXELS`
# ≈ 89.5MP），因此拦截的是"header 报得出尺寸、字节却很小"的超大图（如一张尺寸巨大
# 的纯色 PNG），避免 Image.open→convert→resize 为其瞬时分配数百 MB 内存。prober 认
# 不出尺寸的格式（webp/tiff 等）仍回落到 Pillow 自带的 bomb 保护。由
# `image_exceeds_hard_limit` 消费：命中即 omit（不再 resize、保留原图），因此
# `resize_base64_image_if_oversized` 不再自带这道护栏。
#
# 公开导出：`read_visual_file` 的读入口用同一个 60MP 阈值判定"必须先在沙盒里
# ffmpeg 压缩、失败即报错（不许原图回退）"的超限图 —— 读路径与持久化 omit 共用
# 一个数值来源，读时压掉的图永远不会走到持久化 omit。与 Rust `nexau-rs`
# `OVERSIZED_IMAGE_PIXELS` 保持同步。
OVERSIZED_IMAGE_PIXELS: Final[int] = 60_000_000
_MAX_DECODE_PIXELS: Final[int] = OVERSIZED_IMAGE_PIXELS

# 读路径的文件字节硬门槛（20 MiB）：`read_visual_file` 对超过此大小的图片文件不再
# 直接读原始字节进进程（decode/base64/传输都是负担），而是先在沙盒里用 ffmpeg 压缩
# 后只读回压缩产物；ffmpeg 不可用时报错。与 `MAX_IMAGE_BASE64_BYTES`（持久化入口的
# base64 字符长度 gate，20 MiB base64 ≈ 15 MiB 原始字节）量纲不同：这里量的是磁盘
# 文件字节。与 Rust `nexau-rs` `OVERSIZED_IMAGE_FILE_SIZE_BYTES` 保持同步。
OVERSIZED_IMAGE_FILE_SIZE_BYTES: Final[int] = 20 * 1024 * 1024

# base64 字符串长度硬上限:与读路径的 20 MiB **原始字节** gate 同一量纲
# (ceil(20MiB/3)*4 ≈ 26.7M base64 字符)。此前直接用 20MiB 字符(≈15.7MiB
# 原始字节)比读路径严格 4/3 倍,留下 15.7-20MB 的窗口:读工具 graceful 放行
# 的界内原图在持久化被静默 omit,工具报成功而模型只看到占位符。统一量纲后
# 窗口消失:读侧放行的必不被 omit,超 20MB 的读侧已 fail-closed。超过此长度
# 直接 omit,连 decode 都不做。
MAX_IMAGE_BASE64_BYTES: Final[int] = -(-(20 * 1024 * 1024) // 3) * 4

OVERSIZED_IMAGE_PLACEHOLDER: Final[str] = "image content omitted because it exceeded the supported size limit; use a smaller image"


def floor_even_dimension(value: float) -> int:
    """Floor ``value`` to an even pixel count (minimum 2).

    ffmpeg's default mjpeg pixel format uses 4:2:0 chroma subsampling, which
    requires even dimensions; flooring (never rounding up) keeps every cap
    guarantee intact (an even-floored dimension is <= the exact scaled one).
    Pillow's JPEG encoder tolerates odd dimensions, but we floor-even here too
    so the persist path and the read path land on identical target sizes.
    """
    floored = math.floor(value)
    return max(2, floored - (floored % 2))


# 目标图长边硬上限:Anthropic API 拒收单边 >8000px 的图,且 patch 计费按
# ceil(w/28)*ceil(h/28)——一张 4,000,000×1 的图面积只有 4MP(界内)却要收
# ~14 万 token。因此"需要 resize"的判定和目标尺寸都必须同时管面积与长边。
# 与 Rust `nexau-rs` 同名常量保持同步。
MAX_TARGET_LONG_EDGE: Final[int] = 8000


def area_capped_dimensions(width: int, height: int, max_pixels: int) -> tuple[int, int] | None:
    """Target dimensions for a pixel-area + long-edge cap, or ``None`` if within both.

    Scale = min(area scale, long-edge scale):both dimensions shrink by the
    same factor and floor to even. The ``floor_even_dimension`` 2px minimum
    can inflate a degenerate side (e.g. height 0.004 → 2), silently blowing
    the area bound for extreme aspect ratios — the post-clamp check re-caps
    the governing side so the guarantee ``target_area <= max_pixels`` holds
    for every input, which also keeps both sides far below JPEG's 65,535
    encodable limit.
    """
    pixels = width * height
    long_edge = max(width, height)
    if pixels <= max_pixels and long_edge <= MAX_TARGET_LONG_EDGE:
        return None
    scale = min(1.0, math.sqrt(max_pixels / pixels), MAX_TARGET_LONG_EDGE / long_edge)
    target_w = floor_even_dimension(width * scale)
    target_h = floor_even_dimension(height * scale)
    # min-2 clamp 可能抬高退化边:把主导边按面积回夹,确保面积保证无条件成立。
    if target_w * target_h > max_pixels:
        if target_w >= target_h:
            target_w = floor_even_dimension(max_pixels / target_h)
        else:
            target_h = floor_even_dimension(max_pixels / target_w)
    return target_w, target_h


def _probe_dimensions_from_b64_prefix(b64_data: str) -> tuple[int, int] | None:
    """Probe ``(width, height)`` from only a bounded base64 *prefix*.

    与 ``token_counter._probe_image_dimensions`` 同一手法：只解码
    ``_PROBE_PREFIX_BASE64_CHARS`` 长度的前缀就够读出四种格式的尺寸，无需为了拿
    尺寸而全解码一张多 MB 的图。``None`` 表示前缀内 header 解不出尺寸（未知格式，
    或被截断/超大的 header）——调用方随后做全量解码兜底。
    """
    if not b64_data:
        return None
    prefix_len = min(len(b64_data), _PROBE_PREFIX_BASE64_CHARS)
    # base64 以 4 字符为一组解码；截到 4 的倍数，避免半组切断触发 "invalid length"。
    prefix = b64_data[: prefix_len - (prefix_len % 4)]
    try:
        raw = base64.b64decode(prefix, validate=False)
    except (ValueError, binascii.Error):
        return None
    return probe_dimensions(raw)


def probe_b64_prefix_dimensions(b64_data: str) -> tuple[int, int] | None:
    """Public prefix-probe:从 base64 前缀解析图片尺寸,认不出返回 ``None``。

    持久化侧(``history_list``)用它判定一个无 mime 线索的裸 ``base64`` 字段
    是否真的是图片 —— 认不出的载荷保守跳过,不进 omit/resize。
    """
    return _probe_dimensions_from_b64_prefix(b64_data)


def _is_animated_image(img: Image.Image) -> bool:
    """``img`` 是否为多帧动图（animated GIF / WebP / APNG）。

    用基类 ``seek`` / ``EOFError`` 契约检测，而非 ``n_frames`` / ``is_animated``
    属性：后两者只定义在各 plugin 子类上（GifImageFile / WebPImageFile /
    带 acTL 的 PngImageFile），从声明类型 ``Image.Image`` 上读它们在 pyright strict
    下是属性访问错误，只能靠被禁用的 ``getattr`` 绕过。``seek`` 声明在基类
    ``Image`` 上（真正的序列 plugin 会 override 它），因此这条检测是类型安全的。
    检测后 seek 回 frame 0，使探测无副作用。
    """
    try:
        img.seek(1)
    except EOFError:
        return False
    img.seek(0)
    return True


def image_exceeds_hard_limit(b64_data: str) -> bool:
    """Whether an image must be *omitted* wholesale rather than resized.

    最前置的 size 硬 gate —— 两道判定都在全解码之前完成，专治 decode 炸弹 + 巨大
    payload：

    1. base64 字符串长度 > ``MAX_IMAGE_BASE64_BYTES``（20 MiB）：直接 True，连
       ``base64.b64decode`` 都不做 —— 巨图 decode 后更大、塞进 history payload 本身
       就是负担；
    2. 否则只解码 ``_PROBE_PREFIX_BASE64_CHARS`` 前缀喂 header prober，像素面积 >
       ``_MAX_DECODE_PIXELS``（60 MP）：True，避免 ``Image.open→convert→resize`` 为
       一张"字节小、尺寸巨"的图瞬时分配数百 MB 内存。

    prober 认不出尺寸的格式（webp/tiff 等）第二道判不出，仅靠第一道 base64 长度
    gate 拦截，余下由下游 ``resize_base64_image_if_oversized`` 里 Pillow 自带的
    DecompressionBomb 保护兜底。命中任一道返回 True，调用方应把该图替换成
    ``OVERSIZED_IMAGE_PLACEHOLDER`` 占位；两道都未命中返回 False，调用方照常走
    ``resize_base64_image_if_oversized`` 降采样。
    """
    if len(b64_data) > MAX_IMAGE_BASE64_BYTES:
        logger.warning(
            "Omitting oversized image: base64 length %d chars exceeds hard limit %d; not decoding",
            len(b64_data),
            MAX_IMAGE_BASE64_BYTES,
        )
        return True
    probed = _probe_dimensions_from_b64_prefix(b64_data)
    if probed is None:
        # header prober 认不出的格式(如 SOF 被巨段 EXIF 推出前缀的 JPEG、
        # 罕见 TIFF 排布):用 Pillow 的 lazy open 兜底拿尺寸 —— `Image.open`
        # 只解析 header 不解码像素,不会触发全解码分配。没有它,一张字节小、
        # 声明像素巨大的图会绕过本 gate,直到 `resize_base64_image_if_oversized`
        # 里 Pillow 全解码(其自带 bomb 阈值 ~179MP 远高于本 gate 的 60MP)。
        probed = _pillow_header_dimensions_from_b64_prefix(b64_data)
    if probed is not None and probed[0] * probed[1] > _MAX_DECODE_PIXELS:
        logger.warning(
            "Omitting oversized image: probed %dx%d = %d px exceeds decode guard %d px",
            probed[0],
            probed[1],
            probed[0] * probed[1],
            _MAX_DECODE_PIXELS,
        )
        return True
    return False


def _pillow_header_dimensions_from_b64_prefix(b64_data: str) -> tuple[int, int] | None:
    """Header-only dimension probe via Pillow's lazy ``Image.open``.

    与 ``_probe_dimensions_from_b64_prefix`` 同样只解码 base64 前缀;截断的
    文件 Pillow 读不出 header 时返回 ``None``(保守放行,后续
    ``resize_base64_image_if_oversized`` 里还有 open 后的第二道守卫)。
    """
    if not b64_data:
        return None
    prefix_len = min(len(b64_data), _PROBE_PREFIX_BASE64_CHARS)
    prefix = b64_data[: prefix_len - (prefix_len % 4)]
    try:
        raw = base64.b64decode(prefix, validate=False)
        with Image.open(BytesIO(raw)) as img:
            return img.width, img.height
    except Exception:
        return None


def resize_base64_image_if_oversized(b64_data: str, mime_type: str) -> tuple[str, str] | None:
    """Downscale an over-budget base64 image entirely in memory (Pillow).

    持久化入口的图片降采样：与 ``read_visual_file`` 的默认降采样口径一致 ——
    同一个 ``DEFAULT_IMAGE_MAX_PIXELS`` 面积封顶、同一个 ``area_capped_dimensions``
    目标尺寸公式 —— 但不依赖 sandbox / ffmpeg / 磁盘文件，因此可在
    ``HistoryList`` 持久化路径上对任意来源的图片就地封顶（用户直传、任意工具/
    MCP 返回的图）。

    调用方应先调 ``image_exceeds_hard_limit``：超大图（base64 > 20 MiB 或像素 >
    ``_MAX_DECODE_PIXELS``）走 omit 占位、不进本函数 —— 本函数不再自带 60 MP 护栏，
    对 prober 认得出尺寸的超大图会照常触发全解码降采样。

    Args:
        b64_data: 图片 base64（不含 ``data:`` 前缀）。
        mime_type: 原始 MIME（仅作参考；实际尺寸/格式由 Pillow 从字节解出）。

    Returns:
        ``(new_base64, "image/jpeg")`` 当图片可解码、为静态图、且像素面积超过
        ``DEFAULT_IMAGE_MAX_PIXELS`` 时；否则 ``None`` 表示保持原图不变。返回
        ``None``（幂等 no-op / 原样保留）的情形：base64 无法解码、图片界内、
        动图（多帧，避免拍平丢帧）、解码/降采样失败 —— 调用方一律原样保留原图，
        绝不 fail。
    """
    # 1. 界内快路径（免全解码）：只解码 base64 前缀喂 header prober（PNG/JPEG/GIF/
    #    BMP，与 read_visual_file / token_counter 同一个）。前缀内读得出尺寸且界内
    #    → 直接返回 None，绝不触碰整张大图。每轮 replace_all 重扫已界内图只付一次
    #    前缀解码，恢复/流转的小图不再被反复全解。超大图（含 > 60 MP）由调用方先
    #    调 ``image_exceeds_hard_limit`` omit 掉，不会走到这里；prober 认不出的格式
    #    （webp/tiff 等）落到下面的 Pillow 分支拿真实尺寸。
    probed = _probe_dimensions_from_b64_prefix(b64_data)
    if probed is not None and probed[0] * probed[1] <= DEFAULT_IMAGE_MAX_PIXELS and max(probed) <= MAX_TARGET_LONG_EDGE:
        return None

    # 2. 需要真实解码：图片超预算或 prober 探不出尺寸（未知格式）。
    try:
        raw = base64.b64decode(b64_data)
    except Exception:
        return None

    try:
        with Image.open(BytesIO(raw)) as img:
            # 解码前的最后一道 bomb 守卫:`Image.open` 是 lazy 的,此刻只读了
            # header;超过 60MP 决不继续 convert/resize(那才是全解码分配)。
            # 走到这里说明上游 hard gate 因某种原因没拦住(直接调用方/探测
            # 失败),保留原图返回 None —— 内存威胁在此消除,存储侧仍有
            # base64 长度 gate 兜底。
            if img.width * img.height > _MAX_DECODE_PIXELS:
                logger.warning(
                    "Refusing to decode oversized image (%dx%d) for resize; keeping original bytes",
                    img.width,
                    img.height,
                )
                return None
            # 动图（多帧 GIF/WebP/APNG）不 resize：``convert("RGB")`` + ``save(JPEG)``
            # 会把动图拍平成单帧、不可逆地丢掉动画。保留原图字节，返回 None。
            if _is_animated_image(img):
                logger.warning(
                    "Skipping resize of animated image (%s, %dx%d); keeping original to avoid flattening it to a single frame",
                    img.format or mime_type,
                    img.width,
                    img.height,
                )
                return None
            target = area_capped_dimensions(img.width, img.height, DEFAULT_IMAGE_MAX_PIXELS)
            if target is None:
                return None
            resized = img.convert("RGB").resize(target, Image.Resampling.LANCZOS)
            buffer = BytesIO()
            resized.save(buffer, format="JPEG", quality=85)
    except Exception:
        # graceful：与 read_visual_file 降采样的“降级不 fail”哲学一致 ——
        # 任何解码/降采样异常都保持原图，绝不让 history 写入失败。
        logger.warning("In-memory image resize failed; keeping original image", exc_info=True)
        return None

    return base64.b64encode(buffer.getvalue()).decode("utf-8"), "image/jpeg"


class OversizedInboundImageError(ValueError):
    """User-supplied inbound image exceeds the hard limits — rejected at entry.

    入口 fail-fast(#601):直传方收到明确错误、可压缩后重试;而不是图片进
    history 后被兜底 omit 成占位符无声消失。
    """


def ensure_inbound_images_within_limits(messages: Sequence[object]) -> None:
    """Reject user-supplied messages whose inline images exceed the hard limits.

    与持久化 omit gate 用同一判定(``image_exceeds_hard_limit``:base64 超
    20MiB 原始字节等效,或 header/Pillow 探出 >60MP)。语义差异:入口是
    **拒绝并告知**,持久化是**静默兜底防炸**。只检查 USER 角色消息 —— 工具
    产出的图由读路径强制压缩闸负责,历史消息在此前的写入时已被处理。
    """
    from nexau.core.messages import ImageBlock, Message, Role

    for message_index, message in enumerate(messages):
        if not isinstance(message, Message) or message.role != Role.USER:
            continue
        for block_index, block in enumerate(message.content):
            if not isinstance(block, ImageBlock) or not block.base64:
                continue
            if image_exceeds_hard_limit(block.base64):
                approx_bytes = len(block.base64) * 3 // 4
                raise OversizedInboundImageError(
                    f"Inbound image (message #{message_index + 1}, block #{block_index + 1}) "
                    f"exceeds the hard limits (~{approx_bytes} bytes decoded; limits: "
                    f"20MiB bytes / 60MP pixels). Compress or downscale the image before "
                    f"sending — oversized originals are rejected at entry instead of being "
                    f"silently omitted downstream."
                )
