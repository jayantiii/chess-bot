# test_env.py

from env import ACMChessEnv
import numpy as np

def test_env():
    env = ACMChessEnv(render_mode="ansi")

    # Reset the environment
    obs, info = env.reset(seed=42)
    print("Initial Observation:\n", obs)
    print("Initial Info:\n", info)
    
    assert isinstance(obs, np.ndarray) and obs.shape == (6, 6), "Invalid observation shape"
    assert isinstance(info, dict), "Info must be a dictionary"

    # Try one valid move
    valid_moves = info["valid_moves"]
    if not valid_moves:
        print("No valid moves available.")
        return

    # Convert move format from ((x1, y1), (x2, y2)) to action format
    (from_x, from_y), (to_x, to_y) = valid_moves[0]
    action = (from_y, from_x, to_y, to_x)  # remember: (row, col) mapping

    obs, reward, terminated, truncated, info = env.step(action)

    print("After first move:")
    print("Observation:\n", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:\n", info)

    # Render
    print("Board Render:")
    print(env.render())

    env.close()

if __name__ == "__main__":
    test_env()
