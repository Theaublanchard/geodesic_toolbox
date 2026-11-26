import torch
from torch import Tensor

from .base_class import CoMetric


class PointCarreCoMetric(CoMetric):
    """
    Cometric that is the pointcarre matrix, ie:
    G(x) = 0.25 * diag({1-||x||^2}^2)
    """

    def __init__(self):
        super().__init__()

    def forward(self, q: Tensor) -> Tensor:
        norm_q_sqr = torch.linalg.vector_norm(q, dim=1) ** 2
        scalar = (1 - norm_q_sqr) ** 2
        return 1 / 4 * scalar[:, None, None] * self.eye(q)

    def metric_tensor(self, q: Tensor) -> Tensor:
        norm_q_sqr = torch.linalg.vector_norm(q, dim=1) ** 2
        scalar = 1 / (1 - norm_q_sqr) ** 2
        return 4 * scalar[:, None, None] * self.eye(q)
