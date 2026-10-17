"""Stream-copy and quant_preserve helpers do not import the CUDA extension."""
import importlib.util
import json
import os
import struct

import pytest
import torch

pytestmark = pytest.mark.nogpu

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load(rel):
    path = os.path.join(_ROOT, rel)
    spec = importlib.util.spec_from_file_location(os.path.basename(path), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stream_copy_preserves_dtype_and_isolates_shards(tmp_path):
    stream = load("exllamav3/conversion/stream_copy.py")
    payload = bytes(range(256)) * 4
    src = tmp_path / "src.bin"
    src.write_bytes(payload)
    dest = tmp_path / "out.safetensors"
    n = stream.write_streamed_tensor(str(src), 0, len(payload), "F8_E4M3", [32, 32], str(dest), "layers.1.engram.embed.weight")
    assert n == len(payload)
    blob = dest.read_bytes()
    header_len = struct.unpack_from("<Q", blob, 0)[0]
    header = json.loads(blob[8:8 + header_len].decode())
    entry = header["layers.1.engram.embed.weight"]
    assert entry["dtype"] == "F8_E4M3"
    assert entry["shape"] == [32, 32]
    beg, end = entry["data_offsets"]
    assert blob[8 + header_len + beg:8 + header_len + end] == payload

    class Fake:
        def __init__(self):
            self.tensor_file_map = {"k": str(src)}
            self.file_headers = {"kpath": {}}
        def find_stc(self, key):
            self.file_headers[str(src)] = {
                "k": {"data_offsets": [0, 8], "dtype": "F8_E8M0", "shape": [8]},
                "_header_offset": 4,
            }
            self.tensor_file_map = {"k": str(src)}
            return self
    filename, offset, nbytes, dtype, shape = stream.source_tensor_span(Fake(), "k")
    assert filename == str(src) and offset == 4 and nbytes == 8 and dtype == "F8_E8M0" and shape == [8]

    shards = [[]]
    size = stream.append_extra(shards, 0, "small", 10, 100)
    size = stream.append_extra(shards, size, "huge", stream.STREAM_COPY_MIN_BYTES, 100)
    size = stream.append_extra(shards, size, "after", 10, 100)
    compact = [s for s in shards if s]
    assert compact[0] == ["small"]
    assert compact[1] == [("stream", "huge")]
    assert compact[2] == ["after"]
    assert size == 10


def test_quant_preserve_roundtrip(tmp_path):
    preserve = load("exllamav3/conversion/quant_preserve.py")
    rows = [
        {"dsv41": {"win.0": torch.arange(4).float(), "pre_mix": torch.zeros(1, 2)}},
        {},
        {"dsv41": {"index_k": torch.ones(2, 3)}},
    ]
    path = tmp_path / "nested" / "quant_preserve.safetensors"
    assert preserve.save_preserves(rows, str(path))
    loaded = preserve.load_preserves(str(path), 3)
    assert torch.equal(loaded[0]["dsv41"]["win.0"], rows[0]["dsv41"]["win.0"])
    assert torch.equal(loaded[2]["dsv41"]["index_k"], rows[2]["dsv41"]["index_k"])
    assert loaded[1] == {}
    assert preserve.save_preserves([{}, {}], str(tmp_path / "empty.safetensors")) is False
