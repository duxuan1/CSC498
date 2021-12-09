import gym

if __name__ == "__main__":
    env = gym.make('gym_robot_arm:robot-arm-v1')

    for i_episode in range(20):
        observation = env.reset()
        while 1:
            env.render()
            # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished")
                break
    env.close()