"""Activation caching: run the net over a corpus and stream residuals to disk (🤖).

The SAE trains on individual d_model residual vectors. A board produces 64 of them (one per square),
so caching is: for every FEN capture (64, d_model), pile them up as (N, 64, d_model) on disk, and
later load FLAT as (N*64, d_model) — row r <-> board r // 64, token r % 64.

  * Forward-pass batching lives behind `capture_fn(fens) -> {module: (B, 64, d_model)}`, so one net
    pass can fill several layers' caches at once (the layer sweep).
  * Memory batching: each layer is a preallocated .npy MEMMAP written chunk by chunk, so peak RAM is
    one chunk, not the corpus.
  * Safety: data is written to a temp file and atomically renamed into place, and a sidecar manifest
    (FEN hash, shape, dtype, and a caller-supplied `signature` naming checkpoint + module + encoding)
    is written LAST. A re-run with a matching manifest skips the net entirely; anything else
    regenerates. A crash mid-write never leaves a file that looks complete.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm

SQUARES = 64  # a transformer Leela net has 64 square-tokens per board

# A capture function maps a batch of FENs to residuals: either one (B, 64, d_model) tensor, or a
# {module_name: (B, 64, d_model)} dict when caching several layers from one forward pass.
CaptureFn = Callable[[Sequence[str]], "torch.Tensor | Mapping[str, torch.Tensor]"]


def batched_capture(capture_one: Callable[[str], torch.Tensor]) -> CaptureFn:
    """Adapt a single-FEN capture (fen -> (64, d_model)) into the batched protocol."""
    def _capture(fens: Sequence[str]) -> torch.Tensor:
        return torch.stack([capture_one(fen) for fen in fens], dim=0)   # (B, 64, d_model)
    return _capture


def _manifest(fens: Sequence[str], d_model: int, dtype, signature: str | None) -> dict:
    """Fingerprint everything that changes the bytes on disk. Order matters (row r <-> board r//64)."""
    h = hashlib.sha256()
    for fen in fens:
        h.update(fen.encode("utf-8"))
        h.update(b"\0")  # delimiter so ["ab","c"] and ["a","bc"] don't collide
    return {
        "n": len(fens),
        "squares": SQUARES,
        "d_model": d_model,
        "dtype": np.dtype(dtype).str,
        "fens_sha256": h.hexdigest(),
        "signature": signature,
    }


def _meta_path(out_path: Path) -> Path:
    return out_path.with_suffix(".meta.json")


def _is_cached(out_path: Path, manifest: dict) -> bool:
    meta = _meta_path(out_path)
    if not (out_path.exists() and meta.exists()):
        return False
    try:
        return json.loads(meta.read_text()) == manifest
    except (json.JSONDecodeError, OSError):
        return False   # corrupt / old-format manifest just means "regenerate"


def cache_activations_multi(
    fens: Sequence[str],
    capture_fn: CaptureFn,
    out_paths: Mapping[str, str | Path],
    d_model: int,
    batch_size: int = 256,
    dtype=np.float32,
    force: bool = False,
    signatures: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Capture residuals for every FEN into one (N, 64, d_model) .npy per module, from ONE pass.

    `capture_fn(fen_batch)` must return {module: (B, 64, d_model)} with a key for every module in
    `out_paths`. `signatures[module]` should name checkpoint + module + input encoding.
    """
    n = len(fens)
    if n == 0:
        raise ValueError("cache_activations got an empty FEN corpus")
    out_paths = {k: Path(v) for k, v in out_paths.items()}
    signatures = signatures or {}
    manifests = {k: _manifest(fens, d_model, dtype, signatures.get(k)) for k in out_paths}
    todo = {k: p for k, p in out_paths.items() if force or not _is_cached(p, manifests[k])}
    if not todo:
        return out_paths   # every layer already cached with matching inputs: the net never runs

    tmp_paths, arrays = {}, {}
    for k, p in todo.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        _meta_path(p).unlink(missing_ok=True)          # stale manifest must not outlive its data
        tmp_paths[k] = p.with_name(p.stem + ".partial.npy")
        arrays[k] = open_memmap(tmp_paths[k], mode="w+", dtype=np.dtype(dtype), shape=(n, SQUARES, d_model))

    progress = tqdm(total=n, unit="board", desc="caching activations")
    for start in range(0, n, batch_size):
        batch_fens = fens[start : start + batch_size]
        b = len(batch_fens)
        resid = capture_fn(batch_fens)
        for k, arr in arrays.items():
            r = resid[k]                               # (b, 64, d_model)
            # Shape guard: a wrong site / squeezed batch dim would silently corrupt training.
            if tuple(r.shape) != (b, SQUARES, d_model):
                raise ValueError(f"capture_fn[{k}] returned {tuple(r.shape)}, expected {(b, SQUARES, d_model)}")
            arr[start : start + b] = r.detach().to("cpu", torch.float32).numpy().astype(dtype, copy=False)
        progress.update(b)
    progress.close()

    for arr in arrays.values():
        arr.flush()
    arrays.clear()                                     # release the memmaps before renaming
    for k in todo:
        os.replace(tmp_paths[k], todo[k])              # atomic: readers see old file or complete new one
        _meta_path(todo[k]).write_text(json.dumps(manifests[k], indent=2))   # manifest LAST
    return out_paths


def cache_activations(
    fens: Sequence[str],
    capture_fn: CaptureFn,
    out_path: str | Path,
    d_model: int,
    batch_size: int = 64,
    dtype=np.float32,
    force: bool = False,
    signature: str | None = None,
) -> Path:
    """Single-module convenience wrapper: `capture_fn(fens) -> (B, 64, d_model)` into one .npy."""
    key = "_"
    paths = cache_activations_multi(
        fens, lambda f: {key: capture_fn(f)}, {key: out_path}, d_model,
        batch_size=batch_size, dtype=dtype, force=force, signatures={key: signature},
    )
    return paths[key]


def load_activations(path: str | Path, flatten: bool = True, in_memory: bool = False) -> torch.Tensor:
    """Load a cached (N, 64, d_model) .npy; flatten to (N*64, d_model) by default.

    Default is a lazy copy-on-write memmap (pages come in as touched; in-place ops never write back).
    `in_memory=True` reads the whole file into RAM — much faster for shuffled minibatch training.
    """
    arr = np.load(Path(path)) if in_memory else np.load(Path(path), mmap_mode="c")
    t = torch.from_numpy(arr)
    if flatten:
        t = t.reshape(-1, arr.shape[-1])               # (N*64, d_model) — each square is one example
    return t
