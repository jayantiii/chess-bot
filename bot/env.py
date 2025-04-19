import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.classes.Board import Board

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

    def __init__(self, render_mode=None, width=600, height=600):
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.board = Board(width, height)
        self.current_side = 'white'
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
            'P': 1, 'N': 2, 'B': 3, 'Q': 4,
            'K': 5, 'S': 6
        }

        board_state = self.board.get_board_state()
        for i in range(6):
            for j in range(6):
                cell = board_state[i][j]
                if cell:
                    color, piece = cell[0], cell[1]
                    offset = 0 if color == 'w' else 6
                    obs[i][j] = piece_map.get(piece, 0) + offset

        return obs

    def _get_info(self):
        return {
            "valid_moves": self.board.get_all_valid_moves(self.current_side),
            "turn": self.current_side,
            "is_check": self.board.is_in_check(self.current_side),
            "is_checkmate": self.board.is_in_checkmate(self.current_side),
            "is_draw": self.board.is_in_draw()
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random, self._np_random_seed = np.random.default_rng(seed), seed
        elif self._np_random is None:
            self._np_random, self._np_random_seed = np.random.default_rng(), -1

        # Initialize board
        self.board = Board(self.width, self.height)
        self.current_side = 'white'
        self.done = False

        return self._get_obs(), self._get_info()

    def step(self, action):
        from_pos, to_pos = decode_action(action)
        reward = 0
        done = False

        # Convert coordinates to match Board class's coordinate system
        start_pos = (from_col, from_row)
        end_pos = (to_col, to_row)

        # Perform move using Board class
        success = self.board.handle_move(start_pos, end_pos)

        if (from_pos, to_pos) not in legal_moves:
            # Illegal move: penalize and do random move
            self.board.make_random_move(self.current_player)
            reward = -1
        else:
            # Check if the move resulted in checkmate
            opponent_color = 'black' if self.current_side == 'white' else 'white'
            reward = 1 if self.board.is_in_checkmate(opponent_color) else 0
            terminated = reward > 0
            truncated = self.board.is_in_draw()
            self.done = terminated or truncated
            self.current_side = 'black' if self.current_side == 'white' else 'white'

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "ansi":
            output = ""
            board_state = self.board.get_board_state()
            for row in board_state:
                output += ' '.join(['__' if x == '' else x for x in row]) + "\n"
            return output
        elif self.render_mode == "human":
            print(self.render("ansi"))

    def close(self):
        pass

    @property
    def np_random(self):
        return self._np_random

    @property
    def np_random_seed(self):
        return self._np_random_seed

    @property
    def unwrapped(self):
        return self
