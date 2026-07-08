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

import math
import struct
from typing import Final

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

    Tries each format nexau's own image tools emit (PNG, JPEG, GIF, BMP); an
    unrecognized or malformed header yields ``None``, which callers must treat
    as "size unknown — act conservatively", never as zero.
    """
    return _probe_png_dimensions(data) or _probe_jpeg_dimensions(data) or _probe_gif_dimensions(data) or _probe_bmp_dimensions(data)


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
    # "BM" file header, then the DIB header's width/height sit at a fixed offset
    # across the BITMAPINFOHEADER family (V1/V4/V5) — signed 32-bit LE; a
    # negative height means top-down row order (magnitude is what we need).
    if len(data) < 26 or data[:2] != b"BM":
        return None
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
