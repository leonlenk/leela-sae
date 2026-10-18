"""Tests for the activation cacher (src/data/cache.py, 🤖).

No net needed: we inject a FAKE capture_fn whose values ENCODE (board index, square index) so we
can assert the on-disk layout and the flatten reshape map rows back to the right board+square.
This is the row <-> (board, token) contract feature mining relies on to trace a firing back to a position.
"""

import numpy as np
import torch

from src.data import cache

D_MODEL = 8  # tiny stand-in for the real 768


def fake_capture(fens):
    """Return (B, 64, D_MODEL) where row [b, sq, :] == b*1000 + sq. Lets tests recover (board, sq)."""
    b = len(fens)
    out = torch.zeros(b, cache.SQUARES, D_MODEL)
    for i, fen in enumerate(fens):
        board_id = int(fen)  # tests pass FENs that are just stringified indices "0","1",...
        for sq in range(cache.SQUARES):
            out[i, sq, :] = board_id * 1000 + sq
    return out


def test_cache_writes_grouped_shape_and_loads_flat(tmp_path):
    fens = [str(i) for i in range(5)]                 # 5 "boards"
    path = tmp_path / "acts.npy"

    cache.cache_activations(fens, fake_capture, path, d_model=D_MODEL, batch_size=2)

    # On disk it stays grouped (N, 64, d_model) so we can map firings back to board+square.
    grouped = cache.load_activations(path, flatten=False)
    assert grouped.shape == (5, cache.SQUARES, D_MODEL)

    # Default load is FLAT: each square becomes one independent SAE training example.
    flat = cache.load_activations(path)               # flatten=True
    assert flat.shape == (5 * cache.SQUARES, D_MODEL)


def test_batch_size_not_dividing_n_covers_every_board(tmp_path):
    # 5 boards with batch_size=2 -> chunks of [2,2,1]; the ragged last chunk must still be written.
    fens = [str(i) for i in range(5)]
    path = tmp_path / "acts.npy"
    cache.cache_activations(fens, fake_capture, path, d_model=D_MODEL, batch_size=2)

    flat = cache.load_activations(path)
    # Row r should equal board (r // 64)*1000 + square (r % 64) for EVERY row — no chunk dropped,
    # no row left as the memmap's initial zero. Board 4 (the lone last chunk) is the real test.
    for r in range(flat.shape[0]):
        board, sq = r // cache.SQUARES, r % cache.SQUARES
        assert flat[r, 0].item() == board * 1000 + sq


def test_batched_capture_adapter_matches_one_by_one(tmp_path):
    # batched_capture wraps a B=1 capture (today's hook) into the (list)->(B,64,d) protocol.
    capture_one = lambda fen: fake_capture([fen])[0]   # (64, d_model)
    adapted = cache.batched_capture(capture_one)

    out = adapted(["3", "7"])                          # (2, 64, d_model)
    assert out.shape == (2, cache.SQUARES, D_MODEL)
    assert out[0, 5, 0].item() == 3 * 1000 + 5         # board "3", square 5
    assert out[1, 5, 0].item() == 7 * 1000 + 5         # board "7", square 5


def test_empty_corpus_raises(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        cache.cache_activations([], fake_capture, tmp_path / "x.npy", d_model=D_MODEL)


def counting_capture():
    """Wrap fake_capture but tally how many boards it actually pushed through. Lets a test prove the
    net forward pass was SKIPPED on a cache hit (calls stays 0 on the second run)."""
    state = {"calls": 0}

    def _capture(fens):
        state["calls"] += len(fens)
        return fake_capture(fens)

    return _capture, state


def test_rerun_with_same_inputs_skips_capture(tmp_path):
    # First run captures all 5 boards; second run with identical inputs must reuse the cache and
    # NOT touch capture_fn again — that's the whole point (capturing is the expensive net pass).
    fens = [str(i) for i in range(5)]
    path = tmp_path / "acts.npy"
    capture, state = counting_capture()

    cache.cache_activations(fens, capture, path, d_model=D_MODEL, batch_size=2)
    assert state["calls"] == 5                         # first run did the work

    cache.cache_activations(fens, capture, path, d_model=D_MODEL, batch_size=2)
    assert state["calls"] == 5                         # second run added nothing -> cache hit

    # And the reused data is still correct end-to-end.
    flat = cache.load_activations(path)
    assert flat[64 + 5, 0].item() == 1 * 1000 + 5      # board 1, square 5


def test_force_regenerates_even_on_match(tmp_path):
    fens = [str(i) for i in range(5)]
    path = tmp_path / "acts.npy"
    capture, state = counting_capture()

    cache.cache_activations(fens, capture, path, d_model=D_MODEL, batch_size=2)
    cache.cache_activations(fens, capture, path, d_model=D_MODEL, batch_size=2, force=True)
    assert state["calls"] == 10                        # force ignored the cache and recaptured


def test_changed_fens_regenerate(tmp_path):
    path = tmp_path / "acts.npy"
    capture, state = counting_capture()

    cache.cache_activations([str(i) for i in range(5)], capture, path, d_model=D_MODEL)
    assert state["calls"] == 5
    # Different corpus -> different fingerprint -> must recapture, not hand back the stale 5-board file.
    cache.cache_activations([str(i) for i in range(3)], capture, path, d_model=D_MODEL)
    assert state["calls"] == 8
    assert cache.load_activations(path, flatten=False).shape[0] == 3


def test_changed_signature_regenerate(tmp_path):
    # Same FENs, but a different model/layer signature must NOT reuse the cache: the numbers differ.
    fens = [str(i) for i in range(5)]
    path = tmp_path / "acts.npy"
    capture, state = counting_capture()

    cache.cache_activations(fens, capture, path, d_model=D_MODEL, signature="netA@layer3")
    cache.cache_activations(fens, capture, path, d_model=D_MODEL, signature="netA@layer7")
    assert state["calls"] == 10                        # layer swap -> regenerate


def varying_capture(fens):
    """(B, 64, D) with a value that differs along EVERY axis: board*1000 + square + dim/100, so a
    permutation of the d_model axis (or of squares, or boards) would be caught."""
    b = len(fens)
    board = torch.tensor([int(f) for f in fens], dtype=torch.float32)[:, None, None]
    sq = torch.arange(cache.SQUARES, dtype=torch.float32)[None, :, None]
    dim = torch.arange(D_MODEL, dtype=torch.float32)[None, None, :]
    return (board * 1000 + sq + dim / 100).expand(b, cache.SQUARES, D_MODEL).clone()


def test_multi_module_cache_writes_each_layer_from_one_pass(tmp_path):
    fens = [str(i) for i in range(5)]
    calls = {"n": 0}

    def capture(batch):
        calls["n"] += 1
        x = varying_capture(batch)
        return {"layerA": x, "layerB": -x}             # two "layers" from the same forward pass

    paths = cache.cache_activations_multi(
        fens, capture, {"layerA": tmp_path / "a.npy", "layerB": tmp_path / "b.npy"},
        d_model=D_MODEL, batch_size=2,
    )
    assert calls["n"] == 3                             # ceil(5/2) forward passes total, not per layer
    a = cache.load_activations(paths["layerA"], flatten=False)
    b = cache.load_activations(paths["layerB"], flatten=False)
    assert torch.equal(a, varying_capture(fens)) and torch.equal(b, -varying_capture(fens))


def test_fp16_cache_roundtrips_within_half_precision(tmp_path):
    fens = [str(i) for i in range(3)]
    path = cache.cache_activations(fens, varying_capture, tmp_path / "h.npy", d_model=D_MODEL,
                                   dtype=np.float16)
    loaded = cache.load_activations(path, flatten=False, in_memory=True)
    assert loaded.dtype == torch.float16
    assert torch.allclose(loaded.float(), varying_capture(fens), rtol=1e-3)


def test_crash_mid_capture_leaves_no_complete_looking_cache(tmp_path):
    import pytest
    fens = [str(i) for i in range(6)]
    path = tmp_path / "acts.npy"

    def flaky(batch):
        if "4" in batch:
            raise RuntimeError("simulated crash")
        return varying_capture(batch)

    with pytest.raises(RuntimeError):
        cache.cache_activations(fens, flaky, path, d_model=D_MODEL, batch_size=2)
    # Neither the final .npy nor its manifest exist, so the next run regenerates from scratch.
    assert not path.exists() and not path.with_suffix(".meta.json").exists()
