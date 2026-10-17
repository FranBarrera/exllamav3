"""Checkpoint the per-row calibration bag V4.1 keeps in ``quant_preserve``.

Cross-layer state (delayed mHC pre-mix, shared compressed KV, indexer top-k) does not
live on a module: conversion forwards one module at a time and drops it. The bag has
to ride the job checkpoint or a resumed conversion would attend with empty caches.
"""
from __future__ import annotations

import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def pack_preserves(preserves: list[dict]) -> dict[str, torch.Tensor]:
    packed = {}
    for row, preserve in enumerate(preserves):
        bag = preserve.get("dsv41")
        if not isinstance(bag, dict):
            continue
        for key, value in bag.items():
            if isinstance(value, torch.Tensor):
                packed[f"r{row}.{key}"] = value.detach().cpu().contiguous()
    return packed


def unpack_preserves(n_rows: int, packed: dict[str, torch.Tensor]) -> list[dict]:
    preserves = [{} for _ in range(n_rows)]
    for key, value in packed.items():
        row_s, dot, bag_key = key.partition(".")
        if not dot or not row_s.startswith("r"):
            raise ValueError(f"unexpected quant_preserve key {key}")
        row = int(row_s[1:])
        preserves[row].setdefault("dsv41", {})[bag_key] = value
    return preserves


def save_preserves(preserves: list[dict], path: str) -> bool:
    packed = pack_preserves(preserves)
    if not packed:
        return False
    os.makedirs(os.path.dirname(path), exist_ok = True)
    save_file(packed, path)
    return True


def load_preserves(path: str, n_rows: int) -> list[dict]:
    with safe_open(path, framework = "pt", device = "cpu") as handle:
        packed = {key: handle.get_tensor(key) for key in handle.keys()}
    return unpack_preserves(n_rows, packed)
