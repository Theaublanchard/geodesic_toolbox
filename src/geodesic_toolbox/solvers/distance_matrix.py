"""
Distance matrix utilities for geodesic solvers.
"""

from itertools import product

import torch
from torch import Tensor
from tqdm import tqdm

from .base import GeodesicDistanceSolver


def dst_mat(a: Tensor, b: Tensor, dst_func: GeodesicDistanceSolver) -> Tensor:
    """
    Compute geodesic distances between the points a and b according to the metric g

    Params:
    a : Tensor (b, dim) start points
    b : Tensor (b, dim) end points
    dst_func : GeodesicDistanceSolver, function that computes the geodesic distance between two batch of points

    Output:
    dst : Tensor (b, b) matrix of geodesic distances where dst[i,j] = dst_func(a[i],b[j])
    """
    B, embed_dim = a.shape
    assert b.shape == (
        B,
        embed_dim,
    ), f"Both tensors must have the same shape {(B, embed_dim)=}"

    dst = torch.zeros(B, B, device=a.device)

    # row wise filling
    for i in tqdm(range(B), desc="Computing distance matrix", unit="row"):
        a_batch = a[i].unsqueeze(0).expand(B, -1)
        distances = dst_func(a_batch, b)
        dst[i] = distances

    return dst


def angle_mat(a: Tensor, b: Tensor, dst_func: GeodesicDistanceSolver) -> Tensor:
    """
    Compute geodesic angle between the points a and b according to the metric g

    Params:
    a : Tensor (b, dim) start points
    b : Tensor (b, dim) end points
    dst_func : GeodesicDistanceSolver, function that computes the geodesic distance between two batch of points

    Output:
    dst : Tensor (b, b) matrix of geodesic similarity where dst[i,j] = dst_func.angle(a[i],b[j])
    """
    B, embed_dim = a.shape
    assert b.shape == (
        B,
        embed_dim,
    ), f"Both tensors must have the same shape {(B, embed_dim)=}"

    dst = torch.zeros(B, B, device=a.device)

    # row wise filling
    for i in range(B):
        a_batch = a[i].unsqueeze(0).expand(B, -1)
        distances = dst_func.angle(a_batch, b)
        dst[i] = distances

    return dst


def dst_mat_naive(a: Tensor, b: Tensor, dst_func: GeodesicDistanceSolver) -> Tensor:
    """
    Compute geodesic distances between the points a and b according to the metric g

    Params:
    a : Tensor (b, dim) start points
    b : Tensor (b, dim) end points
    dst_func : GeodesicDistanceSolver, function that computes the geodesic distance between two batch of points

    Output:
    dst : Tensor (b, b) matrix of geodesic distances where dst[i,j] = dst_func(a[i],b[j])
    """
    B, embed_dim = a.shape
    assert b.shape == (
        B,
        embed_dim,
    ), f"Both tensors must have the same shape {(B, embed_dim)=}"

    dst = torch.zeros(B, B, device=a.device)

    for i, j in tqdm(
        product(range(B), range(B)),
        total=B * B,
        desc="Computing distance matrix",
        unit="pair",
    ):
        distances = dst_func(a[i].unsqueeze(0), b[j].unsqueeze(0))
        dst[i, j] = distances

    return dst


def dst_mat_vectorized(
    a: Tensor,
    b: Tensor,
    dst_func: GeodesicDistanceSolver,
) -> Tensor:
    """
    Compute geodesic distances between points a and b efficiently.
    Vectorizes the computation to avoid explicit loops.
    Beware that this is memory intensive (risk of OOM) and might not be faster
    than the naive implementation for small batch sizes.

    Note that the result is not exactly the same as the naive implementation
    because of the optimisation process that optimize for the whole batch at once.

    Params:
    a : Tensor (b, dim) start points
    b : Tensor (b, dim) end points
    dst_func : GeodesicDistanceSolver, function that computes geodesic distances

    Output:
    dst : Tensor (b, b) matrix of geodesic distances where dst[i,j] = dst_func(a[i],b[j])
    """
    # @TODO add automatic routing of method based on batch size

    B, embed_dim = a.shape
    assert b.shape == (
        B,
        embed_dim,
    ), f"Both tensors must have the same shape {(B, embed_dim)=}"

    # Expand dimensions to create all pairs
    # Shape: (B, 1, dim) and (1, B, dim)
    a_expanded = a.unsqueeze(1)  # Shape: (B, 1, dim)
    b_expanded = b.unsqueeze(0)  # Shape: (1, B, dim)

    # Broadcast to (B, B, dim)
    a_tiled = a_expanded.expand(B, B, embed_dim)
    b_tiled = b_expanded.expand(B, B, embed_dim)

    # Reshape to (B*B, dim) for batch processing
    a_flat = a_tiled.reshape(-1, embed_dim)
    b_flat = b_tiled.reshape(-1, embed_dim)

    # Compute distances for all pairs at once
    distances_flat = dst_func(a_flat, b_flat)

    # Reshape back to (B, B)
    distances = distances_flat.reshape(B, B)

    return distances
