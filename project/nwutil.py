"""Network utilities for the Routing and Spectrum Allocation (RSA) environment.

This module defines the network topology, link state helpers, and request
loading utilities shared between the environment and the training script.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------
# Topology and path definition
# ----------------------------

# The four source-destination pairs and their candidate paths.
# Each path is expressed as an ordered list of node ids.
PATH_CATALOG: Dict[str, List[int]] = {
    "P1": [0, 1, 2, 3],
    "P2": [0, 8, 7, 6, 3],
    "P3": [0, 1, 5, 4],
    "P4": [0, 8, 7, 6, 3, 4],
    "P5": [7, 1, 2, 3],
    "P6": [7, 6, 3],
    "P7": [7, 1, 5, 4],
    "P8": [7, 6, 3, 4],
}

# Mapping from request (src, dst) to the pair of path names.
PAIR_TO_PATHS: Dict[Tuple[int, int], Tuple[str, str]] = {
    (0, 3): ("P1", "P2"),
    (0, 4): ("P3", "P4"),
    (7, 3): ("P5", "P6"),
    (7, 4): ("P7", "P8"),
}


def edge_key(u: int, v: int) -> Tuple[int, int]:
    """Return a canonical undirected edge key."""
    return (u, v) if u <= v else (v, u)


def path_to_edges(nodes: Sequence[int]) -> List[Tuple[int, int]]:
    """Convert a node path to a list of canonical edges."""
    return [edge_key(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]


def enumerate_all_edges() -> List[Tuple[int, int]]:
    """Collect the unique set of edges across all predefined paths."""
    edges = set()
    for path_nodes in PATH_CATALOG.values():
        for e in path_to_edges(path_nodes):
            edges.add(e)
    # Sort for deterministic ordering
    return sorted(edges)


# ----------------------------
# Request and link definitions
# ----------------------------

@dataclass
class Request:
    source: int
    destination: int
    holding_time: int


@dataclass
class BaseLinkState:
    endpoints: Tuple[int, int]
    capacity: int
    utilization: int = 0


@dataclass
class LinkState(BaseLinkState):
    """Tracks per-wavelength occupancy on a link."""

    wavelengths: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.wavelengths = np.zeros(self.capacity, dtype=np.int16)

    def first_free_color(self) -> int | None:
        """Return the smallest available wavelength index, if any."""
        free = np.where(self.wavelengths == 0)[0]
        return int(free[0]) if free.size else None

    def occupy(self, color: int, holding_time: int) -> None:
        """Reserve a wavelength for a given holding time (in slots)."""
        self.wavelengths[color] = holding_time
        self.utilization = int(np.count_nonzero(self.wavelengths))

    def tick(self) -> None:
        """Advance time by one slot and release expired wavelengths."""
        self.wavelengths = np.where(self.wavelengths > 0, self.wavelengths - 1, 0)
        self.utilization = int(np.count_nonzero(self.wavelengths))

    def as_binary(self) -> np.ndarray:
        """Binary occupancy vector used for observations."""
        return (self.wavelengths > 0).astype(np.float32)


# ----------------------------
# Graph construction helpers
# ----------------------------

def generate_sample_graph(capacity: int) -> Dict[Tuple[int, int], LinkState]:
    """Create link states for the sample topology with uniform capacity."""
    edges = enumerate_all_edges()
    return {edge: LinkState(edge, capacity) for edge in edges}


def path_edges(path_name: str) -> List[Tuple[int, int]]:
    """Return the edge list for a named path."""
    nodes = PATH_CATALOG[path_name]
    return path_to_edges(nodes)


def first_fit_color(
    path_edges_list: Sequence[Tuple[int, int]],
    link_states: Dict[Tuple[int, int], LinkState],
) -> int | None:
    """Return the smallest wavelength index available on all edges of a path."""
    if not path_edges_list:
        return None
    capacity = link_states[path_edges_list[0]].capacity
    for color in range(capacity):
        if all(link_states[e].wavelengths[color] == 0 for e in path_edges_list):
            return color
    return None


def requests_from_csv(path: Path) -> List[Request]:
    """Load a request sequence from a CSV file."""
    rows: List[Request] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                Request(
                    source=int(row["source"]),
                    destination=int(row["destination"]),
                    holding_time=int(row["holding_time"]),
                )
            )
    return rows


def list_request_files(dataset_dir: Path | str) -> List[Path]:
    """Return sorted CSV files inside a dataset directory."""
    dir_path = Path(dataset_dir)
    return sorted(dir_path.glob("*.csv"))


def occupancy_matrix(
    link_states: Dict[Tuple[int, int], LinkState],
    edge_order: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Stack binary occupancy vectors for all edges in a deterministic order."""
    return np.stack([link_states[e].as_binary() for e in edge_order], axis=0)


def candidate_paths(src: int, dst: int) -> Tuple[str, str]:
    """Return the two candidate path names for a source-destination pair."""
    try:
        return PAIR_TO_PATHS[(src, dst)]
    except KeyError:
        raise ValueError(f"No predefined paths for ({src}, {dst})")


