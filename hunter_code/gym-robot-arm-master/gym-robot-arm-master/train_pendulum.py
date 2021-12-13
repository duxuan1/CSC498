import gym
import numpy as np

from stable_baselines3 import SAC

env = gym.make("Pendulum-v0")

model = SAC("MlpPolicy", env, tensorboard_log="/tmp/sac_pendulum/",verbose=1)
model.learn(total_timesteps=10000, tb_log_name="second_run", log_interval=4)
model.learn(total_timesteps=15000, tb_log_name="second_run", log_interval=4, reset_num_timesteps=False)
model.save("sac_pendulum")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()