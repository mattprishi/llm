"""Ring collectives implemented with point-to-point torch.distributed calls."""

import torch
import torch.distributed as dist


def _ring():
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed process group is not initialized")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, (rank - 1) % world_size, (rank + 1) % world_size


def _check_vector(tensor, name):
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor for the Gloo exercise")


def ring_reduce_scatter(tensor):
    _check_vector(tensor, "tensor")
    rank, world_size, left, right = _ring()
    if tensor.numel() % world_size:
        raise ValueError("tensor length must be divisible by world_size")
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]
    if world_size == 1:
        return chunks[0]

    receive_buffer = torch.empty_like(chunks[0])
    for step in range(world_size - 1):
        send_index = (rank - step - 1) % world_size
        receive_index = (rank - step - 2) % world_size
        receive_request = dist.irecv(receive_buffer, src=left)
        send_request = dist.isend(chunks[send_index], dst=right)
        receive_request.wait()
        send_request.wait()
        chunks[receive_index].add_(receive_buffer)
    return chunks[rank]


def ring_all_gather(shard):
    _check_vector(shard, "shard")
    rank, world_size, left, right = _ring()
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)
    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        receive_index = (rank - step - 1) % world_size
        receive_request = dist.irecv(chunks[receive_index], src=left)
        send_request = dist.isend(chunks[send_index], dst=right)
        receive_request.wait()
        send_request.wait()
    return torch.cat(chunks)


def ring_all_reduce(tensor):
    return ring_all_gather(ring_reduce_scatter(tensor))
