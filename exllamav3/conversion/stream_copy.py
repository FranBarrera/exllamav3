"""Copy one safetensors tensor into its own shard without reading it into RAM.

Engram hash tables are ~90 GiB of FP8 each. compile.py uses this for extras at or
above STREAM_COPY_MIN_BYTES so the output keeps the source dtype string (F8_E4M3,
F8_E8M0) and the conversion process never materializes the table.
"""
from __future__ import annotations

import json
import struct

STREAM_COPY_MIN_BYTES = 256 * 1024 * 1024
_CHUNK = 8 * 1024 * 1024


def should_stream_copy(n_bytes: int) -> bool:
    return n_bytes >= STREAM_COPY_MIN_BYTES


def append_extra(shards: list, current_size: int, key: str, n_bytes: int, max_shard: int) -> int:
    """Place one extra tensor onto the compile shard list.

    A tensor at or above ``STREAM_COPY_MIN_BYTES`` is isolated in its own shard (marked
    with a ``("stream", key)`` entry) even when it is larger than ``max_shard``. The
    caller filters empty shards after the last append.
    """
    if should_stream_copy(n_bytes):
        if shards[-1]:
            shards.append([])
        shards[-1].append(("stream", key))
        shards.append([])
        return 0
    if current_size + n_bytes > max_shard and current_size > 0:
        shards.append([])
        current_size = 0
    shards[-1].append(key)
    return current_size + n_bytes


def write_streamed_tensor(
    src_path: str,
    src_offset: int,
    n_bytes: int,
    dtype: str,
    shape: list,
    dest_path: str,
    key: str,
) -> int:
    """Write a one-tensor safetensors file. Returns the number of payload bytes copied.

    ``dtype`` is the safetensors dtype tag from the source header, not a torch name.
    The header is space-padded to a multiple of 8, which is what the format requires
    so the payload starts 8-byte aligned.
    """
    header = {
        key: {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [0, n_bytes],
        }
    }
    blob = json.dumps(header, separators = (",", ":")).encode("utf-8")
    pad = (8 - (len(blob) % 8)) % 8
    blob += b" " * pad
    remaining = n_bytes
    with open(dest_path, "wb") as out, open(src_path, "rb") as src:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        src.seek(src_offset)
        while remaining:
            chunk = src.read(min(_CHUNK, remaining))
            if not chunk:
                raise IOError(f"short read copying {key} from {src_path}")
            out.write(chunk)
            remaining -= len(chunk)
    return n_bytes


def source_tensor_span(stc, key: str) -> tuple[str, int, int, str, list]:
    """(filename, absolute payload offset, nbytes, dtype tag, shape) for one tensor."""
    base = stc.find_stc(key) if hasattr(stc, "find_stc") else stc
    filename = base.tensor_file_map[key]
    header = base.file_headers[filename]
    entry = header[key]
    beg, end = entry["data_offsets"]
    return filename, header["_header_offset"] + beg, end - beg, entry["dtype"], list(entry["shape"])
