"""Custom Gymnasium environment for the Routing and Spectrum Allocation task."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from nwutil import (
    LinkState,
    PAIR_TO_PATHS,
    candidate_paths,
    first_fit_color,
    generate_sample_graph,
    list_request_files,
    occupancy_matrix,
    path_edges,
    requests_from_csv,
)


class RSAEnv(gym.Env):
    """Routing and Spectrum Allocation environment.

    Observation
        Dict with:
        - "links": binary occupancy matrix of shape (num_links, capacity)
        - "request": [source, destination, holding_time]
    Action
        Discrete(2): choose the first or second predefined path for the current request pair.
    Reward
        +1 for accepted requests, -1 for blocked requests.
    Episode termination
        After all requests in a CSV file are processed (default 100 steps).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset_dir: str | Path,
        capacity: int = 20,
        max_holding_time: int = 100,
        episode_length: int = 100,
        file_selection: str = "random",
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.capacity = capacity
        self.max_holding_time = max_holding_time
        self.episode_length = episode_length
        self.file_selection = file_selection

        self.edge_order: List[Tuple[int, int]] = sorted(generate_sample_graph(capacity).keys())
        self._link_states: Dict[Tuple[int, int], LinkState] = {}
        self.request_files: List[Path] = list_request_files(self.dataset_dir)
        if not self.request_files:
            raise FileNotFoundError(f"No CSV files found in {self.dataset_dir}")

        # Observations: link occupancy and the incoming request tuple
        self.observation_space = spaces.Dict(
            {
                "links": spaces.Box(
                    low=0.0, high=1.0, shape=(len(self.edge_order), capacity), dtype=np.float32
                ),
                "request": spaces.Box(
                    low=np.array([0, 0, 1], dtype=np.int16),
                    high=np.array([8, 8, self.max_holding_time], dtype=np.int16),
                    dtype=np.int16,
                ),
            }
        )
        self.action_space = spaces.Discrete(2)

        self._rng = np.random.default_rng()
        self._file_cursor = 0
        self._requests: List = []
        self._req_idx = 0
        self._current_request = None
        self.blocked = 0
        self.accepted = 0

    # -------------
    # Gym API
    # -------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._link_states = generate_sample_graph(self.capacity)
        self.blocked = 0
        self.accepted = 0
        self._req_idx = 0

        selected_file = self._select_file(options)
        self._requests = requests_from_csv(selected_file)
        if len(self._requests) < self.episode_length:
            raise ValueError(f"File {selected_file} has fewer than {self.episode_length} requests.")
        self._requests = self._requests[: self.episode_length]
        self._current_request = self._requests[self._req_idx]

        observation = self._get_obs()
        info = {"file": selected_file.name}
        return observation, info

    def step(self, action: int):
        assert self.action_space.contains(action)
        req = self._current_request
        blocked = 0

        path_names = candidate_paths(req.source, req.destination)
        chosen_path_name = path_names[action]
        path_edges_list = path_edges(chosen_path_name)
        color = first_fit_color(path_edges_list, self._link_states)

        if color is None:
            reward = -1.0
            blocked = 1
        else:
            reward = 1.0
            for edge in path_edges_list:
                self._link_states[edge].occupy(color, req.holding_time)
            self.accepted += 1

        self.blocked += blocked
        self._advance_time()

        self._req_idx += 1
        terminated = self._req_idx >= self.episode_length
        if not terminated:
            self._current_request = self._requests[self._req_idx]
        else:
            self._current_request = None

        observation = self._get_obs()
        info = {
            "blocked": blocked,
            "chosen_path": chosen_path_name,
            "file_idx": self._req_idx,
        }
        if terminated:
            total = max(self.accepted + self.blocked, 1)
            info["block_rate"] = self.blocked / total

        return observation, reward, terminated, False, info

    # -------------
    # Helpers
    # -------------
    def _advance_time(self) -> None:
        for link in self._link_states.values():
            link.tick()

    def _select_file(self, options: dict | None) -> Path:
        if options and "file_path" in options:
            return Path(options["file_path"])
        if self.file_selection == "sequential":
            path = self.request_files[self._file_cursor % len(self.request_files)]
            self._file_cursor += 1
            return path
        return self._rng.choice(self.request_files)

    def _get_obs(self):
        links = occupancy_matrix(self._link_states, self.edge_order).astype(np.float32)
        if self._current_request is None:
            req_arr = np.array([0, 0, 0], dtype=np.int16)
        else:
            req_arr = np.array(
                [
                    self._current_request.source,
                    self._current_request.destination,
                    self._current_request.holding_time,
                ],
                dtype=np.int16,
            )
        return {"links": links, "request": req_arr}


def make_env(
    dataset_dir: str | Path,
    capacity: int,
    max_holding_time: int = 100,
    episode_length: int = 100,
    seed: int | None = None,
    file_selection: str = "random",
) -> RSAEnv:
    """Factory for convenience and parity with SB3 helpers."""
    env = RSAEnv(
        dataset_dir=dataset_dir,
        capacity=capacity,
        max_holding_time=max_holding_time,
        episode_length=episode_length,
        file_selection=file_selection,
    )
    env.reset(seed=seed)
    return env


