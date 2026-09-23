# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple.npu_mem_ops import (
    DIRECTION_D2H,
    DIRECTION_H2D,
    build_params,
    copy_blocks,
)


class FakeStream:
    def __init__(self) -> None:
        self.sync_count = 0

    def synchronize(self) -> None:
        self.sync_count += 1


class FakeSwapBlocksBatch:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []

    def __call__(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        sizes: torch.Tensor,
        direction: int,
    ) -> None:
        self.calls.append((src, dst, sizes, direction))


@pytest.fixture
def dma(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeStream, FakeSwapBlocksBatch]:
    stream = FakeStream()
    swap = FakeSwapBlocksBatch()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_stream=lambda: stream),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops,
        "_C_ascend",
        SimpleNamespace(swap_blocks_batch=swap),
        raising=False,
    )
    return stream, swap


def _caches(num_blocks: int, block_bytes: int) -> tuple[dict, dict]:
    npu_caches = {
        "layer.0": torch.zeros(num_blocks, block_bytes, dtype=torch.uint8),
        "layer.1": torch.zeros(num_blocks, block_bytes, dtype=torch.uint8),
    }
    cpu_caches = {
        "layer.0": torch.zeros(num_blocks, block_bytes, dtype=torch.uint8),
        "layer.1": torch.zeros(num_blocks, block_bytes, dtype=torch.uint8),
    }
    return npu_caches, cpu_caches


def test_copy_blocks_describes_every_sub_tensor_and_waits_for_the_dma(
    dma: tuple[FakeStream, FakeSwapBlocksBatch],
) -> None:
    stream, swap = dma
    npu_caches, cpu_caches = _caches(num_blocks=8, block_bytes=16)
    params = build_params(npu_caches, cpu_caches, DIRECTION_H2D)

    copy_blocks([1, 3], [5, 7], params)

    ((src, dst, sizes, direction),) = swap.calls
    assert direction == DIRECTION_H2D
    assert isinstance(src, torch.Tensor)
    assert src.tolist() == [
        npu_caches["layer.0"].data_ptr() + 1 * 16,
        npu_caches["layer.0"].data_ptr() + 3 * 16,
        npu_caches["layer.1"].data_ptr() + 1 * 16,
        npu_caches["layer.1"].data_ptr() + 3 * 16,
    ]
    assert dst.tolist() == [
        cpu_caches["layer.0"].data_ptr() + 5 * 16,
        cpu_caches["layer.0"].data_ptr() + 7 * 16,
        cpu_caches["layer.1"].data_ptr() + 5 * 16,
        cpu_caches["layer.1"].data_ptr() + 7 * 16,
    ]
    assert sizes.tolist() == [16] * 4
    # The descriptor arrays live in this frame, so the copy must have completed
    # before copy_blocks returns and releases them.
    assert stream.sync_count == 1


def test_copy_blocks_empty_batch_issues_nothing(
    dma: tuple[FakeStream, FakeSwapBlocksBatch],
) -> None:
    stream, swap = dma
    npu_caches, cpu_caches = _caches(num_blocks=8, block_bytes=16)
    params = build_params(npu_caches, cpu_caches, DIRECTION_H2D)

    copy_blocks([], [], params)

    assert swap.calls == []
    assert stream.sync_count == 0


def test_build_params_records_block_counts_of_each_side() -> None:
    npu_caches, cpu_caches = _caches(num_blocks=8, block_bytes=16)
    cpu_caches["layer.1"] = torch.zeros(5, 16, dtype=torch.uint8)
    params = build_params(npu_caches, cpu_caches, DIRECTION_H2D)

    assert params.src_num_blocks == 8
    assert params.dst_num_blocks == 5


@pytest.mark.parametrize(
    ("src_blocks", "dst_blocks", "message"),
    [
        ([-1], [0], "source block outside"),
        ([8], [0], "source block outside"),
        ([0], [-1], "destination block outside"),
        ([0], [5], "destination block outside"),
    ],
)
def test_copy_blocks_rejects_ids_outside_the_registered_pools(
    dma: tuple[FakeStream, FakeSwapBlocksBatch],
    src_blocks: list[int],
    dst_blocks: list[int],
    message: str,
) -> None:
    stream, swap = dma
    npu_caches, cpu_caches = _caches(num_blocks=8, block_bytes=16)
    cpu_caches["layer.1"] = torch.zeros(5, 16, dtype=torch.uint8)
    params = build_params(npu_caches, cpu_caches, DIRECTION_H2D)

    with pytest.raises(ValueError, match=message):
        copy_blocks(src_blocks, dst_blocks, params)

    # An out-of-range id must never reach the DMA, and nothing was queued for
    # this call either.
    assert swap.calls == []
    assert stream.sync_count == 0


def test_build_params_records_direction_and_per_block_bytes() -> None:
    npu_caches, cpu_caches = _caches(num_blocks=8, block_bytes=16)
    params = build_params(cpu_caches, npu_caches, DIRECTION_D2H)

    assert params.num_sub_tensors == 2
    assert params.direction == DIRECTION_D2H
    assert params.bpb.tolist() == [16, 16]
    assert params.src_bases.tolist() == [c.data_ptr() for c in cpu_caches.values()]
    assert params.dst_bases.tolist() == [c.data_ptr() for c in npu_caches.values()]


def test_build_params_rejects_mismatched_block_bytes() -> None:
    npu_caches, cpu_caches = _caches(num_blocks=8, block_bytes=16)
    cpu_caches["layer.1"] = torch.zeros(8, 32, dtype=torch.uint8)

    with pytest.raises(AssertionError, match="per-block bytes mismatch"):
        build_params(npu_caches, cpu_caches, DIRECTION_H2D)
