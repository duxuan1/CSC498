import gym
import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    print("start train")
    env = gym.make('gym_robot_arm:robot-arm-v1')
    env.seed(0)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    for i in range(5):
        model = TD3("MlpPolicy",
            env,
            action_noise=action_noise,
            tensorboard_log="./td3_robot_test_sample_size/",
            verbose=1
            )
        print("start learn")
        model.learn(total_timesteps=(i+1)*10000, tb_log_name="run_" + str(i))

        model.save('./td3_robot_test_sample_size_models/' + str((i+1)*10000))
    del model

    # Show model performance
    """
    model = SAC.load('sac_robot_arm')
    obs = env.reset()
    for i_episode in range(5):
        episode_reward = 0
        observation = env.reset()
        for t in range(100):
            env.render()
            #print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            # test
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print("Reward:", episode_reward)
                episode_reward = 0.0
                obs = env.reset()
                break
    env.close()

    env = gym.make('gym_robot_arm:robot-arm-v1')
    # try to evaluate policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    """