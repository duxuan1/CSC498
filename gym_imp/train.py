import gym
from stable_baselines import SAC, TD3
from stable_baselines.sac.policies import MlpPolicy

if __name__ == "__main__":
    env = gym.make('gym_robot_arm:robot-arm-v1')

    n_sampled_goal = 4

    model = SAC(MlpPolicy,
                env,
                verbose=1
                )

    model.learn(int(2e5))
    model.save('sac_robot_arm')

    obs = env.reset()

    episode_reward = 0
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        episode_reward += reward

        if done or info.get('is_success', False):
            print("Reward:", episode_reward, "Success?", info.get('is_success', False))
            episode_reward = 0.0
            obs = env.reset()
