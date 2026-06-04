"""Activation caching: run the net over a corpus and stack residuals to disk (agent-fillable 🤖).

The SAE trains on individual d_model residual vectors. A board produces 64 of them (one per
square), so caching is just: for every FEN, capture (64, d_model); pile them up as (N, 64, d_model)
on disk; later load FLAT as (N*64, d_model) — each square is one independent training example.

Two batching axes live here, and they are NOT the same thing:
  * forward-pass batching  — how many boards we push through the net per capture call. This is the
    speed knob. It lives behind `capture_fn`: today's hook (src/hook.py get_residual_stream) is
    B=1, so the default wrapper calls it one FEN at a time. When you write a batched capture in the
    🔒 hook, give it signature (list[str]) -> (B, 64, d_model) and pass it in here unchanged.
  * memory batching        — we never hold all N*64*d_model floats in RAM. We preallocate a .npy
    MEMMAP on disk and write each chunk straight into it, so caching 50k boards costs ~one chunk
    of RAM, not the whole pile.
Flattening is a THIRD, separate thing: it's a reshape of the finished dataset, not a batch size.
Minibatch SGD over the flat rows happens later, in src/train.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm

# A capture function maps a batch of FENs to their residual stream: (B, 64, d_model).
CaptureFn = Callable[[Sequence[str]], torch.Tensor]

SQUARES = 64  # a transformer Leela net has 64 square-tokens per board (DESIGN.md §1)


def batched_capture(capture_one: Callable[[str], torch.Tensor]) -> CaptureFn:
    """Adapt a single-FEN capture (today's B=1 hook.get_residual_stream) into a batch capture.

    Lets cache_activations always speak the batched (list[str]) -> (B, 64, d_model) protocol even
    though src/hook.py only exposes a one-board capture right now. Swap this out for a real batched
    hook later and the cacher doesn't change.
    """
    def _capture(fens: Sequence[str]) -> torch.Tensor:
        # Stack the per-board (64, d_model) captures into (B, 64, d_model).
        per_board = [capture_one(fen) for fen in fens]   # B x (64, d_model)
        return torch.stack(per_board, dim=0)             # (B, 64, d_model)

    return _capture


def _manifest(
    fens: Sequence[str],
    d_model: int,
    dtype: np.dtype | type,
    signature: str | None,
) -> dict:
    """Fingerprint everything that changes the bytes on disk, so a re-run can tell "same inputs?".

    Order matters: row r maps to board r // 64, so a reordered FEN list is a DIFFERENT cache. We
    hash the FENs (cheap, content-addressed) and record shape/dtype. `signature` is the one thing
    we CAN'T see from here — which model + which residual layer produced the activations. Identical
    FENs off a different layer are completely different numbers, so the caller must pass that in or
    a layer swap would silently reuse a stale cache. (See the footgun note in cache_activations.)
    """
    h = hashlib.sha256()
    for fen in fens:
        h.update(fen.encode("utf-8"))
        h.update(b"\0")  # delimiter so ["ab","c"] and ["a","bc"] don't collide
    return {
        "n": len(fens),
        "squares": SQUARES,
        "d_model": d_model,
        "dtype": np.dtype(dtype).str,   # e.g. "<f4" — endianness + width, stable across runs
        "fens_sha256": h.hexdigest(),
        "signature": signature,
    }


def cache_activations(
    fens: Sequence[str],
    capture_fn: CaptureFn,
    out_path: str | Path,
    d_model: int,
    batch_size: int = 64,
    dtype: np.dtype | type = np.float32,
    force: bool = False,
    signature: str | None = None,
) -> Path:
    """Capture residuals for every FEN and stream them to a (N, 64, d_model) .npy memmap on disk.

    `capture_fn(fen_batch) -> (B, 64, d_model)`. We write each batch straight into the on-disk
    memmap so peak RAM is ~one batch, not the whole corpus. The grouped (N, 64, d_model) shape is
    preserved on disk on purpose: it lets you map a firing back to (board #, square) for the
    "believe one feature" check (02_train_small.py TODO 5). Load FLAT for training via
    load_activations(..., flatten=True).

    Caching is EXPENSIVE (a full net forward pass per board), so by default we skip it when a cache
    with matching inputs already sits at out_path. Two safeguards:
      * a sidecar `<out_path>.meta.json` fingerprints the inputs (FENs, shape, dtype, signature);
        a re-run with a matching fingerprint returns immediately without touching the net.
      * `force=True` ignores any existing cache and regenerates unconditionally.
    FOOTGUN: the fingerprint can't see which model/layer produced the activations — pass that via
    `signature` (e.g. f"{model_name}@{residual_module}") or a layer swap will reuse a stale cache.
    """
    out_path = Path(out_path)
    meta_path = out_path.with_suffix(".meta.json")  # sidecar manifest, next to the .npy
    n = len(fens)
    if n == 0:
        raise ValueError("cache_activations got an empty FEN corpus")

    manifest = _manifest(fens, d_model, dtype, signature)

    # Cache hit: reuse iff we weren't told to force AND both files exist AND the fingerprint matches.
    # We read the sidecar defensively — a corrupt/old-format meta just means "regenerate", not crash.
    if not force and out_path.exists() and meta_path.exists():
        try:
            cached = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            cached = None
        if cached == manifest:
            return out_path  # skip the whole capture loop — the net never runs

    # Preallocate the full (N, 64, d_model) array on disk. open_memmap writes a real .npy header,
    # so the shape/dtype travel with the file and np.load(..., mmap_mode='r') can reopen it lazily.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Stale manifest must NOT outlive its data: drop it before we start overwriting the .npy, so a
    # crash mid-capture leaves no manifest -> next run regenerates instead of trusting half a file.
    meta_path.unlink(missing_ok=True)
    arr = open_memmap(out_path, mode="w+", dtype=np.dtype(dtype), shape=(n, SQUARES, d_model))

    # Progress bar over boards (not batches): unit="board" so the rate reads as boards/s, the number
    # you actually care about. tqdm needs the total up front since `range` is consumed lazily.
    progress = tqdm(total=n, unit="board", desc="caching activations")
    for start in range(0, n, batch_size):                 # walk the corpus in chunks of batch_size
        batch_fens = fens[start : start + batch_size]      # last chunk may be < batch_size — fine
        resid = capture_fn(batch_fens)                     # (b, 64, d_model)

        # Shape guard: a wrong activation site / squeezed batch dim would silently corrupt training.
        b = len(batch_fens)
        if tuple(resid.shape) != (b, SQUARES, d_model):
            raise ValueError(
                f"capture_fn returned {tuple(resid.shape)}, expected {(b, SQUARES, d_model)}"
            )

        # Detach off the autograd graph, move to CPU, cast to the on-disk dtype, write the slice.
        arr[start : start + b] = resid.detach().cpu().numpy().astype(dtype, copy=False)

        progress.update(b)  # advance by the boards actually written this chunk
    progress.close()

    arr.flush()  # ensure every batch is on disk before we hand back the path
    # Manifest LAST: only a fully-written .npy gets a fingerprint, so the cache-hit check above can
    # trust that a matching meta.json means the data beside it is complete.
    meta_path.write_text(json.dumps(manifest, indent=2))
    return out_path


def load_activations(path: str | Path, flatten: bool = True) -> torch.Tensor:
    """Load a cached (N, 64, d_model) memmap as a torch tensor; flatten to (N*64, d_model) by default.

    mmap_mode='r' keeps it lazy — the OS pages in rows as the train loop touches them, so a cache
    bigger than RAM still works. torch.from_numpy shares that buffer (no copy); the reshape to
    (N*64, d_model) is a view because the .npy is C-contiguous. Row r ↔ board r // 64, square r % 64.
    """
    # mode='c' = copy-on-write: still lazily paged from disk, but pages become writable on touch
    # WITHOUT writing back to the file. So downstream in-place ops (e.g. activation normalization)
    # are safe and torch gets a writable buffer (no "non-writable tensor" warning).
    arr = np.load(Path(path), mmap_mode="c")     # (N, 64, d_model), lazily paged, copy-on-write
    t = torch.from_numpy(arr)                     # shares the memmap buffer, no full-cache copy
    if flatten:
        t = t.reshape(-1, arr.shape[-1])          # (N*64, d_model) — each square is one example
    return t
