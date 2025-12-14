import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from net_env import NetworkEnv

# Verify the env
def make_env(seed: int=100):
    env = NetworkEnv()
    env = Monitor(env)  # tracks episode rewards/lengths to env.get_episode_rewards(), env.get_episode_lengths()
    env.reset(seed=seed)
    return env

env_for_check = NetworkEnv()
check_env(env_for_check, warn=True)


# Train DQN
SEED = 123
N_EPISODES = 1_000
PLOT_PATH = "learning_curve.png"
MODEL_PATH = "dqn_networkenv_model"

set_random_seed(SEED)
env = make_env(seed=SEED)
print('-------------------')
# Typical DQN config (See the official docs)
model = DQN(
    "MultiInputPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=10_000,
    learning_starts=100,
    batch_size=64,
    tau=0.8,
    gamma=0.99,
    train_freq=4,
    target_update_interval=10_000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    verbose=1,
    seed=SEED,
)

# Use a very large total_timesteps; the callback will stop on episode count.
MAX_HT = 100
env.reset()
model.learn(total_timesteps=N_EPISODES * MAX_HT, progress_bar=True)

model.save(MODEL_PATH)
print(f"Saved model to: {MODEL_PATH}.zip")
print(f"Saved learning curve to: {PLOT_PATH}")

# Cleanly close training env
env.close()


print('-------------------')
# Test run
test_env = NetworkEnv()

obs, info = test_env.reset(seed=SEED + 1)
ep_return = 0.0

# Load back the model to demonstrate persistence (optional)
model = DQN.load(MODEL_PATH)

done = False
while not done:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    ep_return += reward
    done = terminated or truncated
    print(done)

print(f"Test episode return: {ep_return:.3f}")
test_env.close()