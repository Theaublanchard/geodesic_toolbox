"""
Base class for geodesic distance solvers and shared helpers.
"""

import torch
from torch import Tensor

from ..cometric import CoMetric, IdentityCoMetric


def _get_padded_neighbor_indices(
    edge_idx: Tensor,
    batch_size: int,
    target_width: int | None = None,
    pad_value: int = 0,
) -> Tensor:
    """
    Pad a nonzero index list so each row has the same number of neighbors.

    Example with target_width=4 and pad_value=0:
    input edge_idx:
    tensor([[0, 1],
            [0, 12],
            [1, 4],
            [1, 5],
            [1, 65]])

    output padded_edge_idx:
    tensor([[0, 1],
            [0, 12],
            [0, 0],
            [0, 0],
            [1, 4],
            [1, 5],
            [1, 65],
            [1, 0]])

    Parameters:
    edge_idx : Tensor (n_edges, 2)
        Output of nonzero(as_tuple=False) on a batch of adjacency rows.
        Column 0 is the row id inside the batch, column 1 is the neighbor id.
    batch_size : int
        Number of rows in the adjacency batch.
    target_width : int, optional
        If provided, each row is padded to at least target_width entries.
        If some row has more neighbors than target_width, width is increased automatically.
    pad_value : int, optional
        Value used for padded neighbor ids.

    Returns:
    padded_edge_idx : Tensor (batch_size * width, 2)
        Padded pair representation [row_in_batch, neighbor_id].
        It can be reshaped back to (batch_size, width) by taking column 1.
    """
    if edge_idx.numel() == 0:
        counts = torch.zeros(batch_size, dtype=torch.long, device=edge_idx.device)
    else:
        counts = torch.bincount(edge_idx[:, 0], minlength=batch_size)

    width = int(counts.max().item()) if counts.numel() > 0 else 0
    if target_width is not None:
        width = max(width, target_width)

    padded_edge_idx = torch.full(
        (batch_size * width, 2),
        pad_value,
        device=edge_idx.device,
        dtype=torch.long,
    )

    row_template = (
        torch.arange(batch_size, device=edge_idx.device, dtype=torch.long)
        .unsqueeze(1)
        .expand(-1, width)
        .reshape(-1)
    )
    padded_edge_idx[:, 0] = row_template

    if edge_idx.numel() == 0 or width == 0:
        return padded_edge_idx

    row_offsets = torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)[:-1]])
    slot_idx = torch.arange(edge_idx.shape[0], device=edge_idx.device, dtype=torch.long)
    slot_idx = slot_idx - row_offsets.repeat_interleave(counts)

    flat_pos = edge_idx[:, 0] * width + slot_idx
    padded_edge_idx[flat_pos, 1] = edge_idx[:, 1]
    return padded_edge_idx


class GeodesicDistanceSolver(torch.nn.Module):
    """Base class for geodesic distance solvers."""

    def __init__(self, cometric: CoMetric = IdentityCoMetric()):
        super().__init__()
        self.cometric = cometric

    def get_trajectories(self, q0: Tensor, q1: Tensor) -> Tensor:
        """Given the start and end points, compute the geodesic path between the two.

        Params:
        q0 : Tensor (b,d), start points.
        q1 : Tensor (b,d), end points

        Output:
        traj_q : Tensor (b,n_pts,dim), points on the trajectory
        """

    def compute_energy(self, traj_q: Tensor, use_midpoints: bool = True) -> Tensor:
        """Given the trajectory, computes its energy according to the metric

        E = 1/2 int_0^1 g_{q(t)}(dot{q}(t), dot{q}(t)) dt

        Parameters:
        -----------
        traj_q : Tensor (b,n_pts,d)
            Points on the trajectories
        use_midpoints : bool
            Whether to use the midpoints of the segments to compute the energy or not.
            If False, the energy is computed using the starting point of each segment.

        Returns:
        --------
        energies : Tensor, (b,)
            Energies of the trajectories
        """
        dq = torch.diff(traj_q, dim=1)
        q = (traj_q[:, 1:, :] + traj_q[:, :-1, :]) / 2

        energies = [self.cometric.energy(m, seg) for m, seg in zip(q, dq)]
        energies = torch.stack(energies)

        # Add a ReLU to avoid negative energies due to numerical errors
        energies = 0.5 * energies.relu().sum(dim=1)
        return energies

    def compute_distance(self, traj_q: Tensor, use_midpoints: bool = True) -> Tensor:
        """Given the trajectory, computes its length according to the metric

        dst = int_0^1 sqrt{g_{q(t)}(dot{q}(t), dot{q}(t))} dt

        Params :
        traj_q : Tensor (b,n_pts,d)
            Points on the trajectories
        use_midpoints : bool
            Whether to use the midpoints of the segments to compute the distance or not.
            If False, the distance is computed using the starting point of each segment. This is less accurate but faster.

        Output :
        distances : Tensor, (b,), distances of the trajectories
        """
        dq = torch.diff(traj_q, dim=1)
        if use_midpoints:
            q = (traj_q[:, 1:, :] + traj_q[:, :-1, :]) / 2
        else:
            q = traj_q[:, :-1, :]

        # # This version uses way too much memory
        # segments = rearrange(segments, "b n d -> (b n) d")
        # midpoints = rearrange(midpoints, "b n d -> (b n) d")
        # distances = self.cometric.inverse_forward(midpoints, segments)  # (b*n, d)
        # distances = torch.einsum("B d, B d -> B", segments, distances)  # (b*n,)
        # distances = rearrange(distances, "(b n) -> b n", b=traj_q.shape[0])

        # # This version is more memory efficient but slower
        distances = [self.cometric.metric(m, seg) for m, seg in zip(q, dq)]
        distances = torch.stack(distances)

        # Add a ReLU to avoid negative distances due to numerical errors
        distances = distances.relu().sum(dim=1)
        return distances

    def forward(self, q0: Tensor, q1: Tensor) -> Tensor:
        """Given two batch of points q0 and q1, compute the geodesic distance between them.

        Parameters:
        -----------
        q0 : Tensor (B,d)
            The starting points
        q1 : Tensor (B,d)
            The ending points

        Returns:
        --------
        dst : Tensor (B,)
            The estimated geodesic distance between the points
        """
        traj_q = self.get_trajectories(q0, q1)
        dst = self.compute_distance(traj_q)
        return dst
