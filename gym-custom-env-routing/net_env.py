import numpy as np

import gymnasium as gym
from gymnasium import spaces

from enum import IntEnum

# wavelength slot state
class State(IntEnum):
    AVAILABLE = 0
    OCCUPIED = 1

# path identifiers
class PathID(IntEnum):
    PATH0 = 0
    PATH1 = 1

# load each request line by line
def line_reader(file_path):
    """Generator that yields one line at a time from a text file."""
    with open(file_path, 'r') as file:
        for line in file:
            yield line.rstrip('\n')

class NetworkEnv(gym.Env):
    """
    A custom Gym environment for a simple network routing problem.
    """
    def __init__(self, render_mode=None, link_cap: int=5, max_ht: int=100):
        # The capacity of each link
        self.link_cap = link_cap

        # The maximum holding time for requests (to end the simulation)
        self.max_ht = max_ht
        
        # The number of time steps (incremented per request arrival)
        self.round = 0
        
        # Here, observations are dictionaries with the link state and request characterization.
        # (This can be modified to provide more information to the agent if needed.)
        # The link state is represented as a binary vector showing the occupied wavelength slots
        # The request is represented by its holding time (This must be updated for your project.)

        self.observation_space = spaces.Dict(
            {
                "link0": spaces.MultiBinary(self.link_cap),
                "link1": spaces.MultiBinary(self.link_cap),
                "req": spaces.Discrete(self.max_ht)
            }
        )

        # auxiliary data structure to keep track of the link states
        self._linkstates = [np.array([State.AVAILABLE] * self.link_cap, dtype=np.int8), # path 0
                            np.array([State.AVAILABLE] * self.link_cap, dtype=np.int8)] # path 1

        # We have 2 actions, corresponding to each path (direct link)
        NUM_PATHS = 2
        self.action_space = spaces.Discrete(NUM_PATHS)

        self.req_loader = line_reader("requests.txt")

    def _get_obs(self):
        # Since we will need to compute observations both in ``reset`` and
        # ``step``, it is often convenient to have a (private) method ``_get_obs``
        # that translates the environment’s state into an observation. However,
        # this is not mandatory and you may as well compute observations in
        # ``reset`` and ``step`` separately:

        binary_ls0 = (self._linkstates[PathID.PATH0] != 0).astype(np.int8)
        binary_ls1 = (self._linkstates[PathID.PATH1] != 0).astype(np.int8)
        return {"link0": binary_ls0, "link1": binary_ls1, "req": self._req}

    def _get_info(self):
        return {}
    
    def _generate_req(self):
        try:
            return int(next(self.req_loader))
        except StopIteration:
            return None
        
    def _find_available_color(self, link_state):
        # find the first available wavelength slot (color) in the given link state
        # here, we assume a direct link. Slots are not combinatorial.
        for color in range(self.link_cap):
            if link_state[color] == State.AVAILABLE:
                return color
        return -1

    def _remove_requests(self):
        self._linkstates[PathID.PATH0] = np.array([State.AVAILABLE] * self.link_cap, dtype=np.int8)
        self._linkstates[PathID.PATH1] = np.array([State.AVAILABLE] * self.link_cap, dtype=np.int8)

    def reset(self, seed=None):
        # Reset
        # ~~~~~
        #
        # The ``reset`` method will be called to initiate a new episode. You may
        # assume that the ``step`` method will not be called before ``reset`` has
        # been called. Moreover, ``reset`` should be called whenever a done signal
        # has been issued. Users may pass the ``seed`` keyword to ``reset`` to
        # initialize any random number generator that is used by the environment
        # to a deterministic state. It is recommended to use the random number
        # generator ``self.np_random`` that is provided by the environment’s base
        # class, ``gymnasium.Env``. If you only use this RNG, you do not need to
        # worry much about seeding, *but you need to remember to call
        # ``super().reset(seed=seed)``* to make sure that ``gymnasium.Env``
        # correctly seeds the RNG. Once this is done, we can randomly set the
        # state of our environment. In our case, we randomly choose the agent’s
        # location and the random sample target positions, until it does not
        # coincide with the agent’s position.
        #
        # The ``reset`` method should return a tuple of the initial observation
        # and some auxiliary information. We can use the methods ``_get_obs`` and
        # ``_get_info`` that we implemented earlier for that:

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.round = 0

        # remove all requests (recover full capacity)
        self._remove_requests()
        
        # generate the first request for the next episode
        self.req_loader = line_reader("requests.txt")
        self._req = int(self._generate_req())
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def _clock_forward(self):
        self.round += 1
        # Decrease holding times and free up slots if needed
        for path_id in [PathID.PATH0, PathID.PATH1]:
            # where condition to safeguard against negative values
            self._linkstates[path_id] = np.where(self._linkstates[path_id] > 0, self._linkstates[path_id] - 1, 0)

    def step(self, action):
        # Step
        # ~~~~
        #
        # The ``step`` method usually contains most of the logic of your
        # environment. It accepts an ``action``, computes the state of the
        # environment after applying that action and returns the 5-tuple
        # ``(observation, reward, terminated, truncated, info)``. 
        # See `gymnasium.Env.step`. Once the new state of the environment has
        # been computed, we can check whether it is a terminal state and we set
        # ``done`` accordingly.

        # An episode is done iff 100 requests have arrived
        self._clock_forward()
        truncated = (self.round == self.max_ht - 1)

        # the received action tells you which path (or direct link here) was chosen by the agent
        # you need to update the link state accordingly and compute the reward
        assert action in [PathID.PATH0, PathID.PATH1], "Invalid action"

        # Check the designated path (direct link)
        available_color = self._find_available_color(self._linkstates[action])
        if available_color == -1:
            # no available wavelength slot found, request is blocked
            reward = -1
        else:
            # request is accepted
            reward = 1
            curr_req_ht = self._req
            self._linkstates[action][available_color] = curr_req_ht

        # load a new request
        self._req = self._generate_req()            
        observation = self._get_obs()
        info = self._get_info()

        print(f'{self.round}, obs: {observation}')
        
        return observation, reward, False, truncated, info