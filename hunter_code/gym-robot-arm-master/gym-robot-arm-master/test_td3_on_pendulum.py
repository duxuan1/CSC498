import gym
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
env = gym.make('gym_robot_arm:robot-arm-v1')

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
# test
print(n_actions)
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

# test
print(action_noise)

model.learn(total_timesteps=3000, log_interval=1)
model.save("td3_pendulum2")
env = model.get_env()

# test
print("done learning")

del model # remove to demonstrate saving and loading

model = TD3.load("td3_pendulum2")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()