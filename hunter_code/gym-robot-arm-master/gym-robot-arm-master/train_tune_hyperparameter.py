import gym
from stable_baselines3 import SAC, TD3
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    print("start train")
    env = gym.make('gym_robot_arm:robot-arm-v1')
    env.seed(0)
    #n_sampled_goal = 4
    for batch_size in [64, 128, 256]:
        model = SAC(MlpPolicy,
            env,
            tensorboard_log="./sac_robot_test_sample_size/",
            verbose=1,
            batch_size=batch_size
            )
        model.learn(total_timesteps=30000, tb_log_name="run_bs_" + str(batch_size))

        model.save('./sac_robot_test_sample_size_models/batch_size_' + str(batch_size))
    del model