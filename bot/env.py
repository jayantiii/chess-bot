# env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ACMChessEnv(gym.Env):
    """
    Gymnasium environment for ACM Chess (6x6 variant).

    Implements:
    - step()
    - reset()
    - render()
    - close()
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 1
    }

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.board = None
        self.current_side = 'w'
        self.done = False
        self.spec = None

        # Define action and observation space
        # Move: ((from_row, from_col), (to_row, to_col)) => 4 discrete coords in 0-5
        self.action_space = spaces.MultiDiscrete([6, 6, 6, 6])

        # Observation: a 6x6 board with 13 possible values
        self.observation_space = spaces.Box(low=0, high=12, shape=(6, 6), dtype=np.uint8)

        self._np_random = None
        self._np_random_seed = -1

    def _get_obs(self):
        # Convert internal board state to integer-coded array
        obs = np.zeros((6, 6), dtype=np.uint8)
        piece_map = {
            '': 0,
            'P': 1, 'C': 2, 'N': 3, 'B': 4,
            'Q': 5, 'K': 6, 'S': 7, 'J': 8,
        }

        for i in range(6):
            for j in range(6):
                cell = self.board[i][j]
                if cell:
                    color, piece = cell[0], cell[1]
                    offset = 0 if color == 'w' else 6
                    obs[i][j] = piece_map.get(piece, 0) + offset

        return obs

    def _get_info(self):
        return {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random, self._np_random_seed = np.random.default_rng(seed), seed
        elif self._np_random is None:
            self._np_random, self._np_random_seed = np.random.default_rng(), -1

        # Initialize board
        self.board = self._initial_board()
        self.current_side = 'w'
        self.done = False

        return self._get_obs(), self._get_info()

    def step(self, action):
        from_row, from_col, to_row, to_col = action

        # Perform move (implement actual move logic externally)
        success = self._handle_move((from_row, from_col), (to_row, to_col))

        if not success:
            reward = -1
            terminated = True
            truncated = False
            self.done = True
        else:
            reward = 1 if self._is_king_captured('b' if self.current_side == 'w' else 'w') else 0
            terminated = reward > 0
            truncated = False
            self.done = terminated
            self.current_side = 'b' if self.current_side == 'w' else 'w'

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "ansi":
            output = ""
            for row in self.board:
                output += ' '.join(['__' if x == '' else x for x in row]) + "\n"
            return output
        elif self.render_mode == "human":
            print(self.render())

    def close(self):
        pass

    def _initial_board(self):
        return [
            ["bC", "bN", "bQ", "bK", "bB", "bS"],
            ["bP", "bP", "bP", "bP", "bP", "bP"],
            ["", "", "", "", "", ""],
            ["", "", "", "", "", ""],
            ["wP", "wP", "wP", "wP", "wP", "wP"],
            ["wC", "wN", "wQ", "wK", "wB", "wS"]
        ]

    def _handle_move(self, from_pos, to_pos):
        # Dummy implementation — replace with rules + validation
        fx, fy = from_pos
        tx, ty = to_pos

        if not (0 <= fx < 6 and 0 <= fy < 6 and 0 <= tx < 6 and 0 <= ty < 6):
            return False

        piece = self.board[fx][fy]
        if not piece or piece[0] != self.current_side:
            return False

        self.board[tx][ty] = piece
        self.board[fx][fy] = ''
        return True

    def _is_king_captured(self, color):
        for row in self.board:
            for cell in row:
                if cell == f"{color}K":
                    return False
        return True

    @property
    def np_random(self):
        return self._np_random

    @property
    def np_random_seed(self):
        return self._np_random_seed

    @property
    def unwrapped(self):
        return self
