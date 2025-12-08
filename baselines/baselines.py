"""
TSP Baseline Implementations

This file implements baseline algorithms for TSP
1. Christofides algorithm
2. Google's OR-Tools optimization library
3. Concorde TSP Solver -> optimal baseline
4. Supervised learning baseline

Data format: "x1 y1 x2 y2 ... xn yn output tour1 tour2 ... tourn tour1"
"""

import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import numpy.typing as npt

# i had issues w importing stuff, adding checks for availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
    # Set device to CUDA if available
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

try:
    import networkx as nx
    from scipy.spatial.distance import cdist
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

try:
    import elkai
    CONCORDE_AVAILABLE = True
except ImportError:
    CONCORDE_AVAILABLE = False


# ================================ Data Loading ================================

def load_tsp_data(file_path: str) -> List[Tuple[npt.NDArray, List[int]]]:
    """
    Load TSP data from file

    Returns:
        List of (coordinates, optimal_tour)
        - coordinates: np.array of shape (n, 2)
        - optimal_tour: list of node indices forming optimal tour accoridng to previous RL(?) implementation
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' output ')
            if len(parts) != 2:
                continue

            coords_str, tour_str = parts
            coords_list = list(map(float, coords_str.split()))
            coords = np.array(coords_list).reshape(-1, 2)

            tour = list(map(int, tour_str.split()))
            data.append((coords, tour))

    return data


def calculate_tour_length(coords: npt.NDArray, tour: List[int]) -> float:
    """
    Returns the total length of a tour

    Args:
        coords: np.array of shape (n, 2) with node coordinates
        tour: list of node indices (1 indexed -> convert to 0)

    Returns:
        Total length (euclidean dist) of the tour
    """
    length = 0.0
    for i in range(len(tour) - 1):
        # converting to 0 index
        idx1 = tour[i] - 1
        idx2 = tour[i + 1] - 1
        length += np.linalg.norm(coords[idx1] - coords[idx2])
    return length

def coords_to_distance_matrix(coords: npt.NDArray) -> npt.NDArray:
    """ Another way to rep TSP -> convert coordinates to distance matrix"""
    return cdist(coords, coords, metric='euclidean')


# ============================ Christofides Algorithm ==========================

def christofides_tsp(coords: npt.NDArray) -> Tuple[List[int], float]:
    """
    Christofides algorithm implementation for TSP

    Args:
        coords: np.array of shape (n, 2)

    Returns:
        (tour, length) where tour is 1-indexed
    """
    # import working?
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX is required for Christofides algorithm")

    n = len(coords)

    # build graph w/ weights
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            G.add_edge(i, j, weight=dist)

    # Step 1: Find MST
    mst = nx.minimum_spanning_tree(G)

    # Step 2: Find odd-degree vertices
    odd_vertices = [v for v in mst.nodes() if mst.degree(v) % 2 == 1]

    # Step 3: Min weight perfect matching on odd vertices
    odd_subgraph = G.subgraph(odd_vertices).copy()
    matching = nx.min_weight_matching(odd_subgraph)

    # Step 4: Combine MST and matching
    multigraph = nx.MultiGraph(mst)
    multigraph.add_edges_from(matching)

    # Step 5: Find Eulerian circuit
    eulerian_circuit = list(nx.eulerian_circuit(multigraph))

    # Step 6: Convert to Hamiltonian circuit (shortcutting)
    visited = set()
    tour = []
    for u, v in eulerian_circuit:
        if u not in visited:
            tour.append(u)
            visited.add(u)

    tour.append(tour[0])

    # Convert to 1-indexed
    tour_1indexed = [idx + 1 for idx in tour]
    length = calculate_tour_length(coords, tour_1indexed)

    return tour_1indexed, length


# =============================== OR-Tools Solver ==============================

def ortools_tsp(coords: npt.NDArray, time_limit_seconds: int = 5, fast_mode: bool = True) -> Tuple[Optional[List[int]], Optional[float]]:
    """
    Google OR-Tools implementation for TSP

    Args:
        coords: np.array of shape (n, 2)
        time_limit_seconds: time limit for solver  -> bounding this cuz sometimes it takes WAY too long
        fast_mode: if True, use faster strategy with less optimization -> same reason as time limit

    Returns:
        (tour, length) where tour is 1-indexed, or (None, None) if solver fails
    """
    # import working?
    if not ORTOOLS_AVAILABLE:
        raise ImportError("OR-Tools is required for OR-Tools baseline")

    n = len(coords)

    # distance matrix (scaled to integers for OR-Tools)
    dist_matrix = coords_to_distance_matrix(coords)
    dist_matrix_scaled = (dist_matrix * 1000000).astype(np.int64)

    # routing model
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    # distance callback
    def distance_callback(from_index, to_index):
        try:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(dist_matrix_scaled[from_node][to_node])
        except (OverflowError, SystemError):
            # python version compatible?
            return 0

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # search params -> using default
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    if fast_mode:
        # use greedy insertion, skip local search -> i saw this was fast
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        # use automatic strategy, faster than guided local search
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        )
    else:
        # slow but maybe better quality
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )

    search_parameters.time_limit.seconds = time_limit_seconds

    # solve!!
    try:
        solution = routing.SolveWithParameters(search_parameters)
    except (OverflowError, SystemError):
        # python version compatible?
        return None, None

    if not solution:
        return None, None

    # extract tour
    tour = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        tour.append(node + 1)  # Convert to 1-indexed
        index = solution.Value(routing.NextVar(index))

    tour.append(tour[0])

    length = calculate_tour_length(coords, tour)
    return tour, length


# ============================= Concorde Solver ================================

def concorde_tsp(coords: npt.NDArray) -> Tuple[List[int], float]:
    """
    Concorde implementation for TSP (w/ elkai)

    Args:
        coords: np.array of shape (n, 2)

    Returns:
        (tour, length) where tour is 1-indexed
    """
    # import working?
    if not CONCORDE_AVAILABLE:
        raise ImportError("elkai required for Concorde, not installed")

    n = len(coords)

    # distance matrix (scaled to integers for Concorde)
    dist_matrix = coords_to_distance_matrix(coords)
    dist_matrix_scaled = (dist_matrix * 1000000).astype(int).tolist()

    # solve with Concorde (1 line of code wow)
    tour_0indexed = elkai.solve_int_matrix(dist_matrix_scaled)

    # Convert to 1-indexed and close the tour
    tour = [node + 1 for node in tour_0indexed]
    tour.append(tour[0])

    length = calculate_tour_length(coords, tour)
    return tour, length


# =========================== Supervised Baseline ==============================

# import working?
if TORCH_AVAILABLE:
    class TSPDataset(Dataset):
        """PyTorch dataset for TSP problems."""

        def __init__(self, data: List[Tuple[npt.NDArray, List[int]]]):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            coords, tour = self.data[idx]
            return torch.FloatTensor(coords), torch.LongTensor(tour)


    class SupervisedTSPModel(nn.Module):
        """
        Supervised learning baseline for TSP, learns to predict tour node probabilities.
        """

        def __init__(self, input_dim: int = 2, hidden_dim: int = 128):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

            # simple encoder: maps each node to a hidden representation
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

            # pool all nodes to get graph representation
            # o/p: map to per-node scores
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            
            # Move model to CUDA if available
            if DEVICE is not None:
                self.to(DEVICE)

        def forward(self, coords):
            """
            Args:
                coords: (batch_size, seq_len, input_dim)

            Returns:
                logits: (batch_size, seq_len) - scores for each node
            """
            # encode node
            encoded = self.encoder(coords)  # (batch, seq_len, hidden_dim)

            # decode -> logits
            logits = self.decoder(encoded).squeeze(-1)  # (batch, seq_len)

            return logits


    def greedy_decode(model: SupervisedTSPModel, coords: npt.NDArray) -> Tuple[List[int], float]:
        """
        Greedy decoding for tour

        Args:
            model: trained SupervisedTSPModel
            coords: np.array of shape (n, 2)

        Returns:
            (tour, length) where tour is 1-indexed
        """
        model.eval()
        n = len(coords)

        with torch.no_grad():
            coords_tensor = torch.FloatTensor(coords).unsqueeze(0)  # (1, n, 2)
            if DEVICE is not None:
                coords_tensor = coords_tensor.to(DEVICE)

            tour = []
            visited = set()

            # pick highest scoring unvisited node (greedy)
            for _ in range(n):
                logits = model(coords_tensor).squeeze(0)  # (n,)

                # mask visited nodes
                for v in visited:
                    logits[v] = float('-inf')

                # choose best unvisited node
                next_node = logits.argmax().item()
                tour.append(next_node + 1)  # Convert to 1-indexed
                visited.add(next_node)

            tour.append(tour[0])

            length = calculate_tour_length(coords, tour)

        return tour, length
else:
    # error when importing :(
    class TSPDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for supervised baseline")

    class SupervisedTSPModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for supervised baseline")

    def greedy_decode(*args, **kwargs):
        raise ImportError("PyTorch is required for supervised baseline")


# ============================ Evaluation Functions ============================

def evaluate_baselines(
    test_data: List[Tuple[npt.NDArray, List[int]]],
    methods: List[str] = None,
    supervised_model: Optional[Any] = None,
    progress: bool = True,
    ortools_time_limit: int = 5,
    ortools_fast_mode: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Big evaluation function to evaluate ALL baseline methods on dataset.

    Args:
        test_data: list of (coordinates, optimal_tour) tuples
        methods: list of methods to evaluate (christofides, supervised, ortools, concorde, dataset_output)
        supervised_model: trained supervised model (required if 'supervised' in methods)
        progress: true -> print progress
        ortools_time_limit: time limit per instance for OR-Tools (default: 5s for speed)
        ortools_fast_mode: use faster OR-Tools strategy (default: True)

    Returns:
        Dictionary with results for each method, structured as:
        {
            'method_name': {
                'avg_length': float,
                'avg_gap': float,  # % from Concorde 'optimal'
                'avg_ratio': float,  # ratio to Concorde 'optimal' length
                'success_rate': float  # % of successfully solved instances -> for debug, can remove
            }
        }
    """
    if methods is None:
        methods = []
        if TORCH_AVAILABLE and supervised_model is not None:
            methods.append('supervised')
        if NETWORKX_AVAILABLE:
            methods.append('christofides')
        if ORTOOLS_AVAILABLE:
            methods.append('ortools')
        if CONCORDE_AVAILABLE:
            methods.append('concorde')
        methods.append('dataset_output')

    results = {method: {'lengths': [], 'gaps': [], 'ratios': [], 'successes': []}
               for method in methods}

    for i, (coords, dataset_tour) in enumerate(test_data):
        dataset_length = calculate_tour_length(coords, dataset_tour)

        if progress and i % 10 == 0:
            print(f"Processing instance {i}/{len(test_data)}...")

        # compute true optimal if concorde is in methods (for gap calculation)
        true_optimal_length = None
        if 'concorde' in methods:
            try:
                _, true_optimal_length = concorde_tsp(coords)
            except:
                true_optimal_length = None

        # if concorde not available or failed, use dataset as reference
        reference_length = true_optimal_length if true_optimal_length is not None else dataset_length

        # actually run all methods!
        for method in methods:
            try:
                if method == 'supervised':
                    if supervised_model is None:
                        continue
                    tour, length = greedy_decode(supervised_model, coords)
                    success = True
                elif method == 'christofides':
                    tour, length = christofides_tsp(coords)
                    success = True
                elif method == 'ortools':
                    tour, length = ortools_tsp(coords, time_limit_seconds=ortools_time_limit, fast_mode=ortools_fast_mode)
                    success = (tour is not None)
                elif method == 'concorde':
                    # computed above
                    tour, length = None, true_optimal_length
                    success = (length is not None)
                elif method == 'dataset_output':
                    tour, length = dataset_tour, dataset_length
                    success = True
                else:
                    continue

                if success and length is not None:
                    gap = ((length - reference_length) / reference_length) * 100
                    ratio = length / reference_length

                    results[method]['lengths'].append(length)
                    results[method]['gaps'].append(gap)
                    results[method]['ratios'].append(ratio)
                    results[method]['successes'].append(1)
                else:
                    results[method]['successes'].append(0)

            except Exception as e:
                if progress:
                    print(f"Error in {method} on instance {i}: {e}")
                    import traceback
                    traceback.print_exc()
                results[method]['successes'].append(0)

    # calc averages
    summary = {}
    for method in methods:
        if len(results[method]['lengths']) > 0:
            summary[method] = {
                'avg_length': np.mean(results[method]['lengths']),
                'avg_gap': np.mean(results[method]['gaps']),
                'avg_ratio': np.mean(results[method]['ratios']),
                'success_rate': np.mean(results[method]['successes'])
            }
        else:
            summary[method] = {
                'avg_length': float('nan'),
                'avg_gap': float('nan'),
                'avg_ratio': float('nan'),
                'success_rate': 0.0
            }

    return summary


def print_results_table(results: Dict[str, Dict[str, float]], problem_size: int = None):
    """Print evaluation results in a nice table"""
    if problem_size:
        print(f"\n{'='*60}")
        print(f"TSP-{problem_size} Evaluation Results")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"Evaluation Results")
        print(f"{'='*60}")

    print(f"{'Method':<15} {'Avg Length':>12} {'Avg Gap (%)':>12} {'Avg Ratio':>12} {'Success Rate':>12}")
    print(f"{'-'*60}")

    for method, metrics in results.items():
        print(f"{method:<15} {metrics['avg_length']:>12.4f} {metrics['avg_gap']:>12.2f} "
              f"{metrics['avg_ratio']:>12.4f} {metrics['success_rate']:>12.2%}")

    print(f"{'='*60}\n")
