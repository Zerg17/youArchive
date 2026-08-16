"""
Image utilities: check if tile is empty, compress, compute checksum,
and composite multiple tiles onto a shared palette canvas.
"""
import io
import hashlib
from typing import Tuple, Optional

import numpy as np
from PIL import Image

from config import TILE_SIZE


def compress_tile(img: Image.Image) -> bytes:
    """
    Re-compress a tile PNG at compress_level=7 WITHOUT any color conversion.
    Saving the original P-mode image directly is lossless and ~33% smaller
    than the server-sent bytes.
    """
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=7)
    return buf.getvalue()


def compute_checksum(img: Image.Image) -> str:
    """SHA-256 over RGBA pixel data — stable across server-side re-encodings."""
    return hashlib.sha256(img.convert("RGBA").tobytes()).hexdigest()


def check_tile_from_download(
    x: int, y: int, tile_data: bytes
) -> Tuple[bool, Optional[Image.Image], Optional[str]]:
    """
    Process a downloaded tile: check if empty, compute checksum.
    Returns (is_empty, image_or_None, checksum_or_None).
    The returned Image is in its ORIGINAL mode so compress_tile() won't re-quantize.
    """
    try:
        img = Image.open(io.BytesIO(tile_data))
        img_rgba = img.convert("RGBA")
    except Exception:
        return True, None, None

    if img.mode == "P" and img.info.get("transparency") is not None:
        alpha = img_rgba.getchannel("A")
    elif img.mode == "RGBA":
        alpha = img.getchannel("A")
    else:
        alpha = img_rgba.getchannel("A")

    if alpha.getextrema() == (0, 0):
        return True, None, None

    return False, img, compute_checksum(img_rgba)


def composite_tiles_to_png(
    tile_map: dict,
    x1: int, y1: int,
    x2: int, y2: int,
) -> bytes:
    """
    Composite a rectangular region of tiles into a single palette PNG.

    The output palette is built dynamically from the colors actually present
    in the tiles — no hardcoded palette needed. Correct regardless of how
    many colors YouPlace uses or adds in the future.

    Algorithm (all work in palette-index space, no RGBA pixel loops):
      1. Open each tile keeping its original P-mode.
      2. Iterate over palette entries of each tile to collect all colors.
      3. Collect unique (R,G,B) colors into a shared palette
         (index 0 = transparent, 1-N = colors; N ≤ 128 in practice).
      4. For each tile build a 256-entry uint8 LUT: local_idx → global_idx.
      5. Apply LUT to pixel array with a single numpy fancy-index op.
      6. Paste non-transparent pixels onto the canvas.

    Single-tile regions return the stored bytes directly (fastest path).
    """
    num_tiles = (x2 - x1 + 1) * (y2 - y1 + 1)

    if num_tiles == 1:
        data = tile_map.get((x1, y1))
        if data:
            return data
        blank = Image.fromarray(np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8), "P")
        blank.putpalette([0] * 768)
        blank.info["transparency"] = 0
        buf = io.BytesIO()
        blank.save(buf, "PNG", compress_level=7)
        return buf.getvalue()

    tiles: list = []
    for iy in range(y1, y2 + 1):
        for ix in range(x1, x2 + 1):
            data = tile_map.get((ix, iy))
            if not data:
                continue
            img = Image.open(io.BytesIO(data))
            if img.mode != "P":
                img = img.convert("RGBA").quantize(colors=255)
            tiles.append((ix, iy, img))

    if not tiles:
        blank = Image.fromarray(
            np.zeros(((y2 - y1 + 1) * TILE_SIZE, (x2 - x1 + 1) * TILE_SIZE), dtype=np.uint8), "P"
        )
        blank.putpalette([0] * 768)
        blank.info["transparency"] = 0
        buf = io.BytesIO()
        blank.save(buf, "PNG", compress_level=7)
        return buf.getvalue()

    color_to_global: dict = {}   # (r,g,b) → 1-based global index

    for _, _, img in tiles:
        raw_pal = img.getpalette()
        trans   = img.info.get("transparency", None)
        n       = len(raw_pal) // 3
        for i in range(n):
            if i == trans:
                continue
            c = (raw_pal[i * 3], raw_pal[i * 3 + 1], raw_pal[i * 3 + 2])
            if c not in color_to_global:
                color_to_global[c] = len(color_to_global) + 1

    palette_flat = np.zeros(256 * 3, dtype=np.uint8)
    for (r, g, b), idx in color_to_global.items():
        palette_flat[idx * 3: idx * 3 + 3] = [r, g, b]

    height = (y2 - y1 + 1) * TILE_SIZE
    width  = (x2 - x1 + 1) * TILE_SIZE
    canvas = np.zeros((height, width), dtype=np.uint8)

    for ix, iy, img in tiles:
        raw_pal = img.getpalette()
        trans   = img.info.get("transparency", None)
        n       = len(raw_pal) // 3

        lut = np.zeros(256, dtype=np.uint8)
        for i in range(n):
            if i != trans:
                c = (raw_pal[i * 3], raw_pal[i * 3 + 1], raw_pal[i * 3 + 2])
                lut[i] = color_to_global.get(c, 0)

        remapped = lut[np.array(img, dtype=np.uint8)]
        py = (iy - y1) * TILE_SIZE
        px = (ix - x1) * TILE_SIZE
        # np.maximum is much faster than boolean-indexed paste for dense tiles:
        # transparent pixels have global index 0, all opaque pixels have index ≥ 1,
        # so max(canvas, remapped) correctly overwrites only non-transparent pixels.
        np.maximum(
            canvas[py: py + TILE_SIZE, px: px + TILE_SIZE],
            remapped,
            out=canvas[py: py + TILE_SIZE, px: px + TILE_SIZE],
        )

    img_out = Image.fromarray(canvas, "P")
    img_out.putpalette(palette_flat.tolist())
    img_out.info["transparency"] = 0
    buf = io.BytesIO()
    img_out.save(buf, "PNG", compress_level=7)
    return buf.getvalue()
