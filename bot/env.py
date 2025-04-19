import gymnasium as gym
from gymnasium import spaces
import numpy as np
from board import ACMBoard
from utils import get_legal_moves, PieceType

class ACMChessEnv(gym.Env):
    def __init__(self):
        super(ACMChessEnv, self).__init__()
        self.board = ACMBoard()
        self.current_player = 1  # 1 for white, -1 for black

        # Actions: from_pos (6x6) -> to_pos (6x6), so 36x36 = 1296 possible actions
        self.action_space = spaces.Discrete(36 * 36)

        # Observation: we can represent the board as a 6x6xN tensor (one-hot encoding)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6, 6, 17))

    def reset(self):
        self.board.reset()
        self.current_player = 1
        return encode_board(self.board.state, self.current_player)

    def step(self, action):
        from_pos, to_pos = decode_action(action)
        reward = 0
        done = False

        legal_moves = get_legal_moves(self.board, self.current_player)

        if (from_pos, to_pos) not in legal_moves:
            # Illegal move: penalize and do random move
            self.board.make_random_move(self.current_player)
            reward = -1
        else:
            result = self.board.move(from_pos, to_pos)
            if result == "win":
                reward = 1
                done = True
            elif result == "draw":
                reward = 0.5
                done = True

        self.current_player *= -1
        obs = encode_board(self.board.state, self.current_player)
        return obs, reward, done, {}

    def render(self, mode="human"):
        self.board.print()

    def close(self):
        pass
