from env import CovidSEIREnv
from stable_baselines3 import SAC
import numpy as np

max_ep_len = 24
vaccine_schedule = (np.arange(0, max_ep_len) ** 2 * 0.08) * 3_000_000
# vaccine_schedule = [1_000_000_000 / max_ep_len] * max_ep_len
init_state_0 = [0.99, 0.01, 0.0, 0.0]
init_state_1 = [0.8, 0.1, 0.1, 0.0]
init_state_2 = [0.75, 0.1, 0.15, 0.0]
init_states = np.array([init_state_0, init_state_1, init_state_2])

# Values from https://arxiv.org/pdf/2005.12777
beta = 0.41  # [0.33, 0.22, 0.18]
gamma = 0.1  # [0.262, 0.085, 0.087]
sigma = 0.2
population = [700_000_000, 200_000_000, 100_000_000]

env = CovidSEIREnv(
    render_mode="human",
    k=3,
    population=population,
    vaccine_schedule=vaccine_schedule,
    max_steps=24,
    beta=beta,
    gamma=gamma,
    sigma=sigma,
    init_states=init_states,
    normalize_reward=True,
    normalize_obs=True,
    continuous_actions=True,
)

# policy_kwargs = dict(net_arch=dict(pi=[32, 16], qf=[32, 16]))
model = SAC("MlpPolicy", env, verbose=2, device="cuda")
model.learn(total_timesteps=50_000, progress_bar=True)
obs, info = env.reset()
# obs /= np.sum(population)
ep_reward = 0
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    # obs /= np.sum(population)
    print(obs)
    print(action)
    ep_reward += reward
    env.render()
    if done:
        break

print(ep_reward)

env.close()
