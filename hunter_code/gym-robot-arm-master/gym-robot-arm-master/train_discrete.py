import gym
from stable_baselines3 import DQN

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    env = gym.make('gym_robot_arm:robot-arm-v0')

    #n_sampled_goal = 4

    model = DQN("MlpPolicy", env, verbose=1)
    #model.learn(total_timesteps=2e5, log_interval=100)
    #model.save("dqn_robot-arm-v0")

    del model # remove to demonstrate saving and loading

    model = DQN.load("dqn_robot-arm-v0")



    obs = env.reset()
    for i_episode in range(20):
        episode_reward = 0
        observation = env.reset()
        for t in range(100):
            env.render()
            #print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print("Reward:", episode_reward, "Success?")
                episode_reward = 0.0
                obs = env.reset()
                break
    env.close()