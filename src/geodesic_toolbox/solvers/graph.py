"""
Graph-based geodesic solvers (Riemannian and Finsler).
"""

import networkx as nx
import numpy as np
import torch
from networkx import DiGraph, Graph, is_connected
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from tqdm import tqdm

from ..cometric import CoMetric
from ..finsler import FinslerMetric
from .base import GeodesicDistanceSolver


class SolverGraph(GeodesicDistanceSolver):
    """Computes the geodesic distances between points using a KNN-graph

    Parameters
    ----------
    cometric : nn.Module
        The cometric to use to compute the geodesic distances
    data : torch.Tensor (N,D)
        The data points to use to compute the graph
    n_neighbors : int
        The number of neighbors to use
    T : int
        The number of points to use for linear interpolation and returned trajectories
    batch_size : int
        The size of the batch to use for the computation of the graph
    max_data_count : int
        The maximum number of data points to use for the graph. If None, all data points
        are used. If the number of data points is larger than this value, the graph is
        computed on a random subset of the data points to save memory.
    smooth_curve : bool
        If True, smooth the curve with a mean kernel before feeding it to the spline
        when reparametrizing the trajectory.
    """

    def __init__(
        self,
        cometric: CoMetric,
        data: torch.Tensor,
        n_neighbors: int,
        T: int = 100,
        batch_size: int = 64,
        max_data_count: int | None = None,
        smooth_curve: bool = True,
    ) -> None:
        super().__init__(cometric)

        if max_data_count is not None and data.shape[0] > max_data_count:
            indices = torch.randperm(data.shape[0])[:max_data_count]
            data = data[indices]
        self.data = data
        self.n_neighbors = n_neighbors
        self.T = T
        self.b_size = batch_size
        self.smooth_curve = smooth_curve

        self._A, self.knn = self.compute_connectivity_matrix(data, n_neighbors)
        self.W = self.compute_weights(self._A, data, batch_size)

        self.predecessors = self.get_predecessors(self.W)

    def compute_connectivity_matrix(self, data: torch.Tensor, n_neighbors: int):
        """
        Compute the connectivity matrix from the graph.
        If the graph is not weakly connected, we connect the different components by adding edges between the closest points of the different components.

        Parameters:
        data : torch.Tensor (N,D)
            The data points to use to compute the graph
        n_neighbors : int
            The number of neighbors to use for the KNN graph

        Returns:
        A : torch.Tensor (N,N)
            The adjacency matrix of the graph
        knn : NearestNeighbors
            The KNN object used for inference
        """
        # Maybe radius graph is better ?
        knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree", n_jobs=-1)
        knn.fit(data.cpu())

        # Find the Euclidean kNN
        N_data = data.shape[0]
        _, indices = knn.kneighbors(data.cpu())
        indices = indices[:, 1:]  # Remove the point itself, (N_data, k)
        A = torch.zeros((N_data, N_data), dtype=torch.int8)
        batch_idx = torch.arange(N_data).view(-1, 1)
        A[batch_idx, indices] = 1
        if not is_connected(Graph(A.cpu().numpy())):
            print("Adjacency Graph is not connected : connecting ")
            A = self.connect_graph(A)
        # Symetrize the adjacency matrix because why not
        A = torch.maximum(A, A.T)
        return A, knn

    @property
    def A(self):
        return self._A

    def load_adjacency_matrix(self, A: torch.Tensor):
        """
        This can be usefull in a graph settings where the
        connectivity matrix comes from elsewhere and we just
        want to fill in the weights with our metric
        We expect the data to come from the same place with the same ordering
        """
        self._A = A
        self.weakly_connected = is_connected(Graph(A.cpu().numpy()))
        if not self.weakly_connected:
            self._A = self.connect_graph(A)
        self.W = self.compute_weights(self._A, self.data, self.b_size)
        self.predecessors = self.get_predecessors(self.W)

    @torch.no_grad()
    def compute_weights(
        self, A: torch.Tensor, data: torch.Tensor, b_size: int = 64
    ) -> torch.Tensor:
        """
        Given the adjecency matrix, fills it using the metric.

        Parameters:
        A : torch.Tensor (n,n)
            Connectivity matrix
        data : torch.Tensor (n,d)
            The data points
        b_size: int
            The batch size for parallel processing

        Returns
        W : torch.Tensor (n,n)
            The weighted adjacency matrix where W[i,j] approx dst(x_i,x_j)
        """
        t = torch.linspace(0, 1, self.T, device=data.device, dtype=data.dtype).view(1, -1, 1)
        N_data = data.shape[0]
        W = torch.zeros((N_data, N_data), device=data.device, dtype=data.dtype)

        pbar = tqdm(
            range(0, N_data, b_size),
            desc="Initialize Graph",
            unit="batch",
        )
        for start in pbar:
            end = min(start + b_size, N_data)
            batch_idx = torch.arange(start, end)  # (b_size,)
            edge_idx = A[batch_idx].nonzero(as_tuple=False)  # (num_edges_in_batch, 2)
            if edge_idx.numel() == 0:
                continue

            row_idx = batch_idx[edge_idx[:, 0]]
            curr_idx = edge_idx[:, 1]

            p_i = data[row_idx][:, None, :]
            p_j = data[curr_idx][:, None, :]

            linear_traj = p_i + t * (p_j - p_i)  # (n_edges, T, d)
            curve_length = self.compute_distance(linear_traj)

            W[row_idx, curr_idx] = curve_length
            W[curr_idx, row_idx] = curve_length

        # Remove padded assignments introduced by variable-degree batches.
        W = W * A.to(device=W.device, dtype=W.dtype)

        W = W.to(device="cpu")
        W = 1 / 2 * (W + W.T)  # Ensure symmetry

        return W

    def get_cc_connections_idx(self, A, X):
        """
        Given the adjacency matrix A and the data points X,
        returns the indices of the connections between the connected components.

        Parameters:
        A (torch.Tensor): Adjacency matrix of shape (n, n).
        X (torch.Tensor): Data points of shape (n, d).

        Returns:
        idx_correspondence (torch.Tensor) of shape (m, 2):
        Indices of the connections between connected components.
            where idx_correspondence[i, 0] is the index of the first point connecting the i-th component
            and idx_correspondence[i, 1] is the index of the second point connecting the i-th component to its
            closest component.
        """
        # Get connected components from the adjacency matrix W
        G = Graph(A.cpu().numpy())
        cc_list = []
        for c in nx.connected_components(G):
            cc_list.append(torch.tensor(list(c)))

        # minus one because the last cluster
        # will have been compared to all previous ones
        idx_correspondence = torch.zeros(len(cc_list) - 1, 2, dtype=torch.long)

        for i in range(len(cc_list) - 1):
            bst_dst_i = float("inf")
            idx_i, idx_j = None, None
            X_cc_i = X[cc_list[i]]
            for j in range(i + 1, len(cc_list)):
                X_cc_j = X[cc_list[j]]

                dst_ij = torch.cdist(X_cc_i, X_cc_j, p=2)
                bst_dst = torch.min(dst_ij)
                if bst_dst < bst_dst_i:
                    bst_dst_i = bst_dst
                    bst_pts = torch.argmin(dst_ij)
                    idx_i, idx_j = torch.unravel_index(bst_pts, dst_ij.shape)
                    # Map idx_i and idx_j to the original indices
                    idx_i = cc_list[i][idx_i]
                    idx_j = cc_list[j][idx_j]
            idx_correspondence[i, 0] = idx_i
            idx_correspondence[i, 1] = idx_j
        return idx_correspondence

    def connect_graph(self, A: torch.Tensor) -> torch.Tensor:
        """
        Connect the connected components of the graph by adding dummy edges.
        This is a workaround to ensure that the graph is connected.
        The cc are connected via their closest points according to euclidean distance.

        Parameters:
        A (torch.Tensor):
            Adjacency matrix of shape (n, n).

        Returns:
        A (torch.Tensor):
            Adjacency matrix of shape (n, n) with added edges to connect the connected components.
        """
        idx_correspondence = self.get_cc_connections_idx(A, self.data)
        # a = self.data[idx_correspondence]
        # t = torch.linspace(0, 1, self.T, device=self.data.device, dtype=W.dtype).view(1, -1, 1)

        # p_i = a[:, 0][:, None, :]
        # p_j = a[:, 1][:, None, :]
        # linear_traj = p_i + t * (p_j - p_i)  # (n_cc,T,d)
        # dst = self.compute_distance(linear_traj).to(W.dtype)
        # W[idx_correspondence[:, 0], idx_correspondence[:, 1]] = dst
        # W[idx_correspondence[:, 1], idx_correspondence[:, 0]] = dst
        A[idx_correspondence[:, 0], idx_correspondence[:, 1]] = 1
        A[idx_correspondence[:, 1], idx_correspondence[:, 0]] = 1
        return A
        # return W

    def get_similarity_matrix(self, W, sigma=1):
        """Compute the similarity matrix using the weight matrix W

        Parameters
        ----------
        W : torch.Tensor (N,N)
            The weight matrix
        sigma : float
            The sigma to use for the similarity matrix

        Returns
        -------
        S : torch.Tensor (N,N)
            The similarity matrix
        """
        S = torch.exp(-W / (2 * sigma**2))
        return S

    def get_predecessors(self, W) -> torch.Tensor:
        print("Computing predecessors...")
        # Use the adjacency matrix A to define which entries are edges,
        # even if their weight in W is 0.
        A_np = self.A.cpu().numpy()
        W_np = W.cpu().numpy()

        # Get indices of all edges defined in A
        rows, cols = np.where(A_np == 1)
        # Get the weights for these specific edges from W
        weights = W_np[rows, cols]

        # Construct CSR matrix explicitly with these edges
        W_csr = csr_matrix((weights, (rows, cols)), shape=W.shape)

        _dst_matrix, predecessors = shortest_path(
            W_csr,
            directed=False,
            return_predecessors=True,
        )
        print("Done.")
        return torch.from_numpy(predecessors)

    ## This version is actually even slower than the one above fuck this
    # def get_predecessors(self,W,device,censor_value=-9999) -> torch.Tensor:
    #     """
    #     Constructs a predecessors matrix from the paths dictionary.
    #     The predecessors matrix is a square matrix where the entry (i, j) contains the predecessor
    #     of node j on the shortest path from node i. If there is no path from i to j, the entry is censor_value.

    #     Parameters:
    #     W (torch.Tensor): The weight matrix of shape (n, n).
    #     device (torch.device): The device on which the computation is performed.
    #     censor_value (int): The value to use for entries where there is no path from i to j.

    #     Returns:
    #     np.ndarray: A 2D numpy array of shape (n, n) containing the predecessors.
    #     """
    #     G = nx.from_numpy_array(W.cpu().numpy())

    #     if device.type == "cuda":
    #         with nx.config.backends.cugraph(n_jobs=4):
    #             paths = nx.all_pairs_dijkstra(G, weight="weight")
    #     elif device.type == "cpu":
    #         with nx.config.backends.parallel(n_jobs=4):
    #             paths = nx.all_pairs_dijkstra(G, weight="weight")

    #     n = G.number_of_nodes()
    #     pred = np.full((n, n), censor_value, dtype=int)
    #     for node_i, (dict_distance_i, dict_path_i) in paths:  # Fixed variable name
    #         for node_j, paths_ij in dict_path_i.items():
    #             if len(paths_ij) > 1:
    #                 pred[node_i, node_j] = paths_ij[-2]
    #     return torch.from_numpy(pred)

    def linear_interpolation(self, p_i, p_j, t):
        """
        Linear interpolation between two points

        Parameters:
        ----------
        p_i : Tensor (b,d)
            The starting point
        p_j : Tensor (b,d)
            The ending point
        t : Tensor (n_pts,)

        Returns:
        --------
        traj_q : Tensor (b,n_pts,d)
            The linear interpolation between the two points
        """
        return p_i[:, None, :] + t[None, :, None] * (p_j - p_i)[:, None, :]

    def connect_to_graph(
        self, q0: Tensor, q1: Tensor, euclidean_only: bool = True
    ) -> tuple[Tensor, Tensor]:
        """
        Given two batch of points q0 and q1, find their closest point in the graph
        The closest points are determined by the linear interpolation metric
        distance (not euclidean) between the points.

        Parameters:
        -----------
        q0 : Tensor (B,d)
            start point
        q1 : Tensor (B,d)
            end point
        euclidean_only : bool
            If True, only use the euclidean distance to find the closest points

        Returns:
        --------
        closest_q0 : Tensor (B,)
            Indicices of the closest point in self.data to q0
        closest_q1 : Tensor (B,)
            Indicices of the closest point in self.data to q1
        """
        B = q0.shape[0]

        # Get the indices of the closest points in the graph
        _, ind0 = self.knn.kneighbors(q0.detach().cpu())
        _, ind1 = self.knn.kneighbors(q1.detach().cpu())

        if euclidean_only:
            closest_q0 = torch.from_numpy(ind0[:, 0])
            closest_q1 = torch.from_numpy(ind1[:, 0])
            return closest_q0.int(), closest_q1.int()

        # Get rid of the last one that we added for initialisation
        ind0 = ind0[:, :-1]  # (B,K)
        ind1 = ind1[:, :-1]

        q0_neighbors = torch.zeros(
            B, self.n_neighbors, self.data.shape[1], device=q0.device, dtype=q0.dtype
        )  # (B,K,D)
        q1_neighbors = torch.zeros(
            B, self.n_neighbors, self.data.shape[1], device=q0.device, dtype=q0.dtype
        )

        for n in range(self.n_neighbors):
            q0_neighbors[:, n, :] = self.data[ind0[:, n]]
            q1_neighbors[:, n, :] = self.data[ind1[:, n]]

        # Find the points with the closest metric distance to q0 and q1
        t = torch.linspace(0, 1, self.T, device=q0.device, dtype=q0.dtype)

        closest_q0 = torch.zeros(B).int()
        closest_q1 = torch.zeros(B).int()

        for b in range(B):
            q0b = q0[b].reshape(1, -1).expand(self.n_neighbors, -1)  # (K,d)
            q0b_neigh = q0_neighbors[b]  # (K,d)
            linear_traj = self.linear_interpolation(
                q0b, q0b_neigh, t
            )  # (n_neighbors, n_pts, d)
            dst_to_q0b = self.compute_distance(linear_traj)  # (n_neighbors,)
            argmin_length = torch.argmin(dst_to_q0b)
            closest_q0[b] = ind0[b, argmin_length]

            q1b = q1[b].reshape(1, -1).expand(self.n_neighbors, -1)  # (K,d)
            q1b_neigh = q1_neighbors[b]  # (K,d)
            linear_traj = self.linear_interpolation(
                q1b, q1b_neigh, t
            )  # (n_neighbors, n_pts, d)
            dst_to_q1b = self.compute_distance(linear_traj)  # (n_neighbors,)
            argmin_length = torch.argmin(dst_to_q1b)
            closest_q1[b] = ind1[b, argmin_length]
        return closest_q0, closest_q1

    def get_path_idx(self, start_idx: Tensor, end_idx: Tensor, max_path_length: int = 2000):
        """
        Given the start and end indices, retrieve the path in the graph.

        Parameters:
        -----------
        start_idx : Tensor (B,)
            The starting indices
        end_idx : Tensor (B,)
            The ending indices
        max_path_length : int
            The maximum length of the path to retrieve for security reasons.

        Returns:
        --------
        path : Tensor (B,max_traj_length)
            The path in the graph. The value -9999 means the path is done.
        """
        B = start_idx.shape[0]
        # Clone to avoid in-place updates corrupting the recorded initial endpoint ids.
        temp_idx = end_idx.clone()
        path = [end_idx.clone()]

        iter = 0
        nb_pts_in_path = torch.zeros(B)
        # Simply properly retrieve the path
        while (nb_pts_in_path == 0).any() and iter < max_path_length:
            iter += 1
            pred_idx = self.predecessors[start_idx, temp_idx]
            if (pred_idx == -9999).any():
                # A path is done
                b_end_path = torch.argwhere(pred_idx == -9999).squeeze(1)
                b_end_path = torch.zeros(B, dtype=bool).scatter(0, b_end_path, True)
                lenght_to_update = (nb_pts_in_path == 0) & b_end_path
                nb_pts_in_path[lenght_to_update] = iter
                temp_idx[b_end_path] = start_idx[b_end_path]
                temp_idx[~b_end_path] = pred_idx[~b_end_path]
            else:
                temp_idx = pred_idx
            path.append(pred_idx.clone())

        path = torch.stack(path, dim=1)
        return path

    def get_pts_from_idx(self, start_idx: Tensor, path_idx: Tensor, censor_value: int = -9999):
        """From the indices, retrieve the points in the graph

        Parameters:
        -----------
        start_idx : Tensor (B,)
            The starting indices of the path.
        path_idx : Tensor (B,max_traj_length)
            The indices of the points in the graph
        censor_value : int
            The value that indicates the end of the path.

        Returns:
        --------
        path : Tensor (B,max_traj_length,d)
            The points in the graph. The end of the path are padded to max_traj_length with the last valid point.
        """
        maks = path_idx == censor_value
        replacement_values = start_idx.unsqueeze(1).expand(-1, path_idx.shape[1])
        # We replace the censor_value by the starting point
        path_idx = torch.where(maks, replacement_values, path_idx)
        pts_on_traj = self.data[path_idx]
        return pts_on_traj

    def get_trajectories(self, q0: Tensor, q1: Tensor, connect_euclidean: bool = False):
        """
        Compute the geodesic trajectories between two points.
        This amounts to properly connecting the points to the KNN graph and then
        computing the shortest path between the two points.

        Parameters:
        -----------
        q0 : Tensor (b,d)
            The starting points
        q1 : Tensor (b,d)
            The ending points
        connect_euclidean : bool
            If True, only use the euclidean distance to find the closest points

        Returns:
        --------
        pts_on_traj : Tensor (b,T,d)
            The points on the trajectory
        """
        start_idx, end_idx = self.connect_to_graph(q0, q1, euclidean_only=connect_euclidean)
        path_idx = self.get_path_idx(start_idx, end_idx)

        pts_on_traj = torch.zeros(
            q0.shape[0], self.T, q0.shape[1], device=q0.device, dtype=q0.dtype
        )
        for b in range(q0.shape[0]):
            pts_on_traj[b] = self.reparametrize_curve(
                path_idx[b], q0[b], q1[b], smooth_curve=self.smooth_curve
            )

        # Reverse end->start to start->end and keep exactly T points.
        pts_on_traj = pts_on_traj.flip(1)
        return pts_on_traj

    def reparametrize_curve(
        self, idx_traj: Tensor, q0, q1, censor_value=-9999, smooth_curve=True
    ) -> Tensor:
        """
        Reparametrize a single curve to have a fixed number of points using a cubic spline.

        Parameters:
        -----------
        idx_traj : Tensor (n_pts)
            The indices of the points on the trajectory.
        q0 : Tensor (d,)
            The starting point
        q1 : Tensor (d,)
            The ending point
        censor_value : int
            The value that indicates the end of the path in idx_traj
        smooth_curve : bool
            If True, smooth the curve with a mean kernel before feeding it to the spline

        Returns:
        --------
        traj_q : Tensor (T,d)
            The reparametrized trajectories
        """
        # Find the first occurence of the censor value
        mask = idx_traj == censor_value
        first_censor = torch.argmax(mask.int())
        # Retrieve the correct trajectory
        idx_traj = idx_traj[:first_censor]
        traj_q = self.data[idx_traj]
        traj_q = torch.cat([q1.unsqueeze(0), traj_q, q0.unsqueeze(0)], dim=0)

        if smooth_curve:
            # Smooth the curve with a mean kernel
            # Pad the curve with two  points at the start and end
            p0 = traj_q[0].unsqueeze(0)
            p1 = traj_q[-1].unsqueeze(0)
            traj_q = torch.cat([p0, p0, traj_q, p1, p1], dim=0)
            traj_q = traj_q.unfold(0, 3, 1).mean(dim=2)

        # Resample the curve
        cs = CubicSpline(np.linspace(0, 1, traj_q.shape[0]), traj_q.detach().cpu())
        traj_q = cs(np.linspace(0, 1, self.T))
        traj_q = torch.from_numpy(traj_q).to(q0.device).to(q0.dtype)
        return traj_q


class SolverGraphFinsler(torch.nn.Module):
    """Computes the geodesic distances between points using a KNN-graph
    for metrics from a Finsler space.

    Parameters
    ----------
    finsler_metric : FinslerMetrics
        The metric to use to compute the geodesic distances
    data : torch.Tensor (N,D)
        The data points used to compute the graph
    n_neighbors : int
        The number of neighbors to use
    T : int
        The number of points to use for the linear interpolation
    batch_size : int
        The size of the batch to use for the computation of the graph
    max_data_count : int
        The maximum number of data points to use for the graph. If None, all data points
        are used. If the number of data points is larger than this value, the graph is
        computed on a random subset of the data points to save memory.
    smooth_curve : bool
        If True, smooth the curve with a mean kernel before feeding it to the spline
        when reparametrizing the trajectory.
    """

    def __init__(
        self,
        finsler_metric: FinslerMetric,
        data: torch.Tensor,
        n_neighbors: int,
        T: int = 100,
        batch_size: int = 64,
        max_data_count: int | None = None,
        smooth_curve: bool = True,
    ) -> None:
        super().__init__()
        self.finsler_metric = finsler_metric

        if max_data_count is not None and data.shape[0] > max_data_count:
            indices = torch.randperm(data.shape[0])[:max_data_count]
            data = data[indices]
        self.data = data
        self.n_neighbors = n_neighbors
        self.T = T
        self.b_size = batch_size
        self.smooth_curve = smooth_curve

        # self.W, self.knn = self.init_knn_graph(data, n_neighbors, batch_size)
        # self.predecessors = self.get_predecessors()

        self._A, self.knn = self.compute_connectivity_matrix(data, n_neighbors)
        self.W = self.compute_weights(self._A, data, batch_size)

        self.predecessors = self.get_predecessors(self.W)

    def compute_connectivity_matrix(self, data: torch.Tensor, n_neighbors: int):
        """
        Compute the connectivity matrix from the graph.
        If the graph is not weakly connected, we connect the different components by adding edges between the closest points of the different components.

        Parameters:
        data : torch.Tensor (N,D)
            The data points to use to compute the graph
        n_neighbors : int
            The number of neighbors to use for the KNN graph

        Returns:
        A : torch.Tensor (N,N)
            The adjacency matrix of the graph
        knn : NearestNeighbors
            The KNN object used for inference
        """
        # Maybe radius graph is better ?
        knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree", n_jobs=-1)
        knn.fit(data.cpu())

        # Find the Euclidean kNN
        N_data = data.shape[0]
        _, indices = knn.kneighbors(data.cpu())
        indices = indices[:, 1:]  # Remove the point itself, (N_data, k)
        A = np.zeros((N_data, N_data), dtype=int)
        batch_idx = torch.arange(N_data).view(-1, 1)
        A[batch_idx, indices] = 1
        if not is_connected(Graph(A)):
            print("Adjacency Graph is not connected : connecting ")
            A = self.connect_graph(A)
        # Symmetrize the adjacency matrix because why not
        A = np.maximum(A, A.T)
        return torch.from_numpy(A), knn

    @property
    def A(self):
        return self._A

    def load_adjacency_matrix(self, A: torch.Tensor):
        """Load an adjacency matrix from an external source and compute the weights and predecessors.
        This can be useful in a graph settings where the connectivity matrix comes from elsewhere and we just
        want to fill in the weights with our metric. We expect the data to come from the same place with the same ordering.

        Parameters:
        A : torch.Tensor (n,n)
            The adjacency matrix to load
        """
        print("Loading adjacency matrix...")
        self._A = A
        self.weakly_connected = is_connected(Graph(A.cpu().numpy()))
        if not self.weakly_connected:
            print("Reconnecting the new mat")
            self._A = self.connect_graph(A)
        print("Recomputing weights")
        self.W = self.compute_weights(self._A, self.data, self.b_size)
        print("Computing precedecessors")
        self.predecessors = self.get_predecessors(self.W)
        print("Over.")

    @torch.no_grad()
    def compute_weights(
        self, A: torch.Tensor, data: torch.Tensor, b_size: int = 64
    ) -> torch.Tensor:
        """
        Given the adjecency matrix, fills it using the metric.

        Parameters:
        A : torch.Tensor (n,n)
            Connectivity matrix
        data : torch.Tensor (n,d)
            The data points
        b_size: int
            The batch size for parallel processing

        Returns
        W : torch.Tensor (n,n)
            The weighted adjacency matrix where W[i,j] approx dst(x_i,x_j)
        """
        t = torch.linspace(0, 1, self.T, device=data.device, dtype=data.dtype).view(1, -1, 1)
        N_data = data.shape[0]
        W = torch.zeros((N_data, N_data), device=data.device, dtype=data.dtype)

        pbar = tqdm(
            range(0, N_data, b_size),
            desc="Initialize Graph",
            unit="batch",
        )
        for start in pbar:
            end = min(start + b_size, N_data)
            batch_idx = torch.arange(start, end)

            edge_idx = A[batch_idx].nonzero(as_tuple=False)
            if edge_idx.numel() == 0:
                continue

            row_idx = batch_idx[edge_idx[:, 0]]
            curr_idx = edge_idx[:, 1]

            p_i = data[row_idx][:, None, :]
            p_j = data[curr_idx][:, None, :]

            linear_traj = p_i + t * (p_j - p_i)  # (n_edges, T, d)
            curve_length = self.compute_distance(linear_traj)

            W[row_idx, curr_idx] = curve_length

        # Remove padded assignments introduced by variable-degree batches.
        W = W * A.to(device=W.device, dtype=W.dtype)

        W = W.to(device="cpu")

        return W

    def get_cc_connections_idx(self, A, X):
        """
        Given the adjacency matrix A and the data points X,
        returns the indices of the connections between the connected components.

        Parameters:
        A (torch.Tensor): Adjacency matrix of shape (n, n).
        X (torch.Tensor): Data points of shape (n, d).

        Returns:
        idx_correspondence (torch.Tensor) of shape (m, 2):
        Indices of the connections between connected components.
            where idx_correspondence[i, 0] is the index of the first point connecting the i-th component
            and idx_correspondence[i, 1] is the index of the second point connecting the i-th component to its
            closest component.
        """
        # Get connected components from the adjacency matrix A
        if isinstance(A, torch.Tensor):
            G = DiGraph(A.cpu().numpy())
        else:
            G = DiGraph(A)
        cc_list = []
        for c in nx.strongly_connected_components(G):
            cc_list.append(torch.tensor(list(c)))

        # minus one because the last cluster
        # will have been compared to all previous ones
        idx_correspondence = torch.zeros(len(cc_list) - 1, 2, dtype=torch.long)

        for i in range(len(cc_list) - 1):
            bst_dst_i = float("inf")
            idx_i, idx_j = None, None
            X_cc_i = X[cc_list[i]]
            for j in range(i + 1, len(cc_list)):
                X_cc_j = X[cc_list[j]]

                dst_ij = torch.cdist(X_cc_i, X_cc_j, p=2)
                bst_dst = torch.min(dst_ij)
                if bst_dst < bst_dst_i:
                    bst_dst_i = bst_dst
                    bst_pts = torch.argmin(dst_ij)
                    idx_i, idx_j = torch.unravel_index(bst_pts, dst_ij.shape)
                    # Map idx_i and idx_j to the original indices
                    idx_i = cc_list[i][idx_i]
                    idx_j = cc_list[j][idx_j]
            idx_correspondence[i, 0] = idx_i
            idx_correspondence[i, 1] = idx_j
        return idx_correspondence

    def connect_graph(self, A: torch.Tensor) -> torch.Tensor:
        """
        Connect the connected components of the graph by adding dummy edges.
        This is a workaround to ensure that the graph is connected.
        The cc are connected via their closest points according to euclidean distance.

        Parameters:
        A (torch.Tensor):
            Adjacency matrix of shape (n, n).

        Returns:
        A (torch.Tensor):
            Adjacency matrix of shape (n, n) with added edges to connect the connected components.
        """
        idx_correspondence = self.get_cc_connections_idx(A, self.data)
        A[idx_correspondence[:, 0], idx_correspondence[:, 1]] = 1
        A[idx_correspondence[:, 1], idx_correspondence[:, 0]] = 1
        return A

    def compute_distance(self, traj: torch.Tensor, tangent_vectors: torch.Tensor = None):
        """Given a trajectory and the tangent vectors, compute the distance
        under the finsler metric.

        Parameters
        ----------
        traj : torch.Tensor (b,T,d)
            The trajectory. There are b trajectories of T points in d dimensions
        tangent_vectors : torch.Tensor (b,T,d)
            The tangent vectors at each point of the trajectory
            If None, the tangent vectors are computed as the difference between consecutive points.
        Returns
        -------
        dst : torch.Tensor (b,)
            The distance between the two points
        """
        if tangent_vectors is None:
            # Compute the tangent vectors as the difference between consecutive points
            tangent_vectors = torch.zeros_like(traj)
            tangent_vectors[:, :-1, :] = traj[:, 1:, :] - traj[:, :-1, :]
            tangent_vectors[:, -1, :] = traj[:, -1, :] - traj[:, -2, :]
        distances = torch.stack(
            [self.finsler_metric(m, seg) for m, seg in zip(traj, tangent_vectors)]
        )  # (B, T)
        distances = distances.relu().sum(dim=1)  # (B,)
        return distances

    def get_similarity_matrix(self, W, sigma=1):
        """Compute the similarity matrix using the weight matrix W

        Parameters
        ----------
        W : torch.Tensor (N,N)
            The weight matrix
        sigma : float
            The sigma to use for the similarity matrix

        Returns
        -------
        S : torch.Tensor (N,N)
            The similarity matrix
        """
        S = torch.exp(-W / (2 * sigma**2))
        return S

    def get_predecessors(self, W) -> torch.Tensor:
        print("Computing predecessors...")
        # Use the adjacency matrix A to define which entries are edges,
        # even if their weight in W is 0.
        A_np = self.A.cpu().numpy()
        W_np = W.cpu().numpy()

        # Get indices of all edges defined in A
        rows, cols = np.where(A_np == 1)
        # Get the weights for these specific edges from W
        weights = W_np[rows, cols]

        # Construct CSR matrix explicitly with these edges
        W_csr = csr_matrix((weights, (rows, cols)), shape=W.shape)

        _dst_matrix, predecessors = shortest_path(
            W_csr,
            directed=True,
            return_predecessors=True,
        )
        print("Done.")
        return torch.from_numpy(predecessors)

    def linear_interpolation(self, p_i, p_j, t):
        """
        Linear interpolation between two points

        Parameters:
        ----------
        p_i : Tensor (b,d)
            The starting point
        p_j : Tensor (b,d)
            The ending point
        t : Tensor (n_pts,)

        Returns:
        --------
        traj_q : Tensor (b,n_pts,d)
            The linear interpolation between the two points
        """
        return p_i[:, None, :] + t[None, :, None] * (p_j - p_i)[:, None, :]

    def connect_to_graph(
        self, q0: Tensor, q1: Tensor, euclidean_only: bool = True
    ) -> tuple[Tensor, Tensor]:
        """
        Given two batch of points q0 and q1, find their closest point in the graph
        The closest points are determined by the linear interpolation metric
        distance (not euclidean) between the points.

        Parameters:
        -----------
        q0 : Tensor (B,d)
            start point
        q1 : Tensor (B,d)
            end point
        euclidean_only : bool
            If True, only use the euclidean distance to find the closest points

        Returns:
        --------
        closest_q0 : Tensor (B,)
            Indicices of the closest point in self.data to q0
        closest_q1 : Tensor (B,)
            Indicices of the closest point in self.data to q1
        """
        B = q0.shape[0]

        # Get the indices of the closest points in the graph
        _, ind0 = self.knn.kneighbors(q0.detach().cpu())
        _, ind1 = self.knn.kneighbors(q1.detach().cpu())

        if euclidean_only:
            closest_q0 = torch.from_numpy(ind0[:, 0])
            closest_q1 = torch.from_numpy(ind1[:, 0])
            return closest_q0.int(), closest_q1.int()

        # Get rid of the last one that we added for initialisation
        ind0 = ind0[:, :-1]  # (B,K)
        ind1 = ind1[:, :-1]

        q0_neighbors = torch.zeros(
            B, self.n_neighbors, self.data.shape[1], device=q0.device, dtype=q0.dtype
        )  # (B,K,D)
        q1_neighbors = torch.zeros(
            B, self.n_neighbors, self.data.shape[1], device=q0.device, dtype=q0.dtype
        )

        for n in range(self.n_neighbors):
            q0_neighbors[:, n, :] = self.data[ind0[:, n]]
            q1_neighbors[:, n, :] = self.data[ind1[:, n]]

        # Find the points with the closest metric distance to q0 and q1
        t = torch.linspace(0, 1, self.T, device=q0.device, dtype=q0.dtype)

        closest_q0 = torch.zeros(B).int()
        closest_q1 = torch.zeros(B).int()

        for b in range(B):
            q0b = q0[b].reshape(1, -1).expand(self.n_neighbors, -1)  # (K,d)
            q0b_neigh = q0_neighbors[b]  # (K,d)
            linear_traj = self.linear_interpolation(
                q0b, q0b_neigh, t
            )  # (n_neighbors, n_pts, d)
            tangent_vectors = (q0b_neigh - q0b)[:, None, :].repeat(
                1, self.T, 1
            )  # (n_neighbors, n_pts, d)
            dst_to_q0b = self.compute_distance(linear_traj, tangent_vectors)  # (n_neighbors,)
            argmin_length = torch.argmin(dst_to_q0b)
            closest_q0[b] = ind0[b, argmin_length]

            q1b = q1[b].reshape(1, -1).expand(self.n_neighbors, -1)  # (K,d)
            q1b_neigh = q1_neighbors[b]  # (K,d)
            linear_traj = self.linear_interpolation(
                q1b, q1b_neigh, t
            )  # (n_neighbors, n_pts, d)
            tangent_vectors = (q1b_neigh - q1b)[:, None, :].repeat(
                1, self.T, 1
            )  # (n_neighbors, n_pts, d)
            dst_to_q1b = self.compute_distance(linear_traj, tangent_vectors)  # (n_neighbors,)
            argmin_length = torch.argmin(dst_to_q1b)
            closest_q1[b] = ind1[b, argmin_length]
        return closest_q0, closest_q1

    def get_path_idx(self, start_idx: Tensor, end_idx: Tensor, max_path_length: int = 2000):
        """
        Given the start and end indices, retrieve the path in the graph.

        Parameters:
        -----------
        start_idx : Tensor (B,)
            The starting indices
        end_idx : Tensor (B,)
            The ending indices
        max_path_length : int
            The maximum length of the path. If the path is longer, it will be truncated.

        Returns:
        --------
        path : Tensor (B,max_traj_length)
            The path in the graph. The value -9999 means the path is done.
        """
        B = start_idx.shape[0]
        # Clone to avoid in-place updates corrupting the recorded initial endpoint ids.
        temp_idx = end_idx.clone()
        path = [end_idx.clone()]

        iter = 0
        nb_pts_in_path = torch.zeros(B)
        # Simply properly retrieve the path
        while (nb_pts_in_path == 0).any() and iter < max_path_length:
            iter += 1
            pred_idx = self.predecessors[start_idx, temp_idx]
            if (pred_idx == -9999).any():
                # A path is done
                b_end_path = torch.argwhere(pred_idx == -9999).squeeze(1)
                b_end_path = torch.zeros(B, dtype=bool).scatter(0, b_end_path, True)
                lenght_to_update = (nb_pts_in_path == 0) & b_end_path
                nb_pts_in_path[lenght_to_update] = iter
                temp_idx[b_end_path] = start_idx[b_end_path]
                temp_idx[~b_end_path] = pred_idx[~b_end_path]
            else:
                temp_idx = pred_idx
            path.append(pred_idx.clone())

            if iter >= max_path_length:
                print("Max path length reached")
                break

        path = torch.stack(path, dim=1)
        return path

    def get_pts_from_idx(self, start_idx: Tensor, path_idx: Tensor, censor_value: int = -9999):
        """From the indices, retrieve the points in the graph

        Parameters:
        -----------
        start_idx : Tensor (B,)
            The starting indices of the path.
        path_idx : Tensor (B,max_traj_length)
            The indices of the points in the graph
        censor_value : int
            The value that indicates the end of the path.

        Returns:
        --------
        path : Tensor (B,max_traj_length,d)
            The points in the graph. The end of the path are padded to max_traj_length with the last valid point.
        """
        maks = path_idx == censor_value
        replacement_values = start_idx.unsqueeze(1).expand(-1, path_idx.shape[1])
        # We replace the censor_value by the starting point
        path_idx = torch.where(maks, replacement_values, path_idx)
        pts_on_traj = self.data[path_idx]
        return pts_on_traj

    def get_trajectories(self, q0: Tensor, q1: Tensor, connect_euclidean: bool = False):
        """
        Compute the geodesic trajectories between two points.
        This amounts to properly connecting the points to the KNN graph and then
        computing the shortest path between the two points.

        Parameters:
        -----------
        q0 : Tensor (b,d)
            The starting points
        q1 : Tensor (b,d)
            The ending points
        connect_euclidean : bool
            If True, only use the euclidean distance to find the closest points

        Returns:
        --------
        pts_on_traj : Tensor (b,T,d)
            The points on the trajectory
        """
        start_idx, end_idx = self.connect_to_graph(q0, q1, euclidean_only=connect_euclidean)
        path_idx = self.get_path_idx(start_idx, end_idx)

        pts_on_traj = torch.zeros(
            q0.shape[0], self.T, q0.shape[1], device=q0.device, dtype=q0.dtype
        )
        for b in range(q0.shape[0]):
            pts_on_traj[b] = self.reparametrize_curve(
                path_idx[b], q0[b], q1[b], smooth_curve=self.smooth_curve
            )

        # Reverse end->start to start->end and keep exactly T points.
        pts_on_traj = pts_on_traj.flip(1)
        return pts_on_traj

    def reparametrize_curve(
        self, idx_traj: Tensor, q0, q1, censor_value=-9999, smooth_curve=True
    ) -> Tensor:
        """
        Reparametrize a single curve to have a fixed number of points using a cubic spline.

        Parameters:
        -----------
        idx_traj : Tensor (n_pts)
            The indices of the points on the trajectory.
        q0 : Tensor (d,)
            The starting point
        q1 : Tensor (d,)
            The ending point
        censor_value : int
            The value that indicates the end of the path in idx_traj
        smooth_curve : bool
            If True, smooth the curve with a mean kernel before feeding it to the spline

        Returns:
        --------
        traj_q : Tensor (T,d)
            The reparametrized trajectories
        """
        # Find the first occurence of the censor value
        mask = idx_traj == censor_value
        first_censor = torch.argmax(mask.int())
        # Retrieve the correct trajectory
        idx_traj = idx_traj[:first_censor]
        traj_q = self.data[idx_traj]
        traj_q = torch.cat([q1.unsqueeze(0), traj_q, q0.unsqueeze(0)], dim=0)

        if smooth_curve:
            # Smooth the curve with a mean kernel
            # Pad the curve with two  points at the start and end
            p0 = traj_q[0].unsqueeze(0)
            p1 = traj_q[-1].unsqueeze(0)
            traj_q = torch.cat([p0, p0, traj_q, p1, p1], dim=0)
            traj_q = traj_q.unfold(0, 3, 1).mean(dim=2)

        # Resample the curve
        cs = CubicSpline(np.linspace(0, 1, traj_q.shape[0]), traj_q.detach().cpu())
        traj_q = cs(np.linspace(0, 1, self.T))
        traj_q = torch.from_numpy(traj_q).float().to(q0.device)
        return traj_q

    def forward(self, q0: Tensor, q1: Tensor, connect_euclidean: bool = False):
        """
        Computes the geodesic distances between two points q0 and q1 using the KNN graph.

        Parameters:
        -----------
        q0 : Tensor (b,d)
            The starting points
        q1 : Tensor (b,d)
            The ending points
        connect_euclidean : bool
            If True, only use the euclidean distance to find the closest points

        Returns:
        --------
        dst : Tensor (b,)
            The geodesic distances between the two points
        """
        pts_on_traj = self.get_trajectories(q0, q1, connect_euclidean=connect_euclidean)
        traj_front = pts_on_traj[:, 1:, :]
        traj_back = pts_on_traj[:, :-1, :]
        segments = traj_front - traj_back
        midpoints = (traj_front + traj_back) / 2

        dst = self.compute_distance(midpoints, segments)

        return dst
