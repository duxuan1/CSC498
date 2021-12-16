# CSC498

This is a OpenAI gym environment for two links robot arm in 2D based on PyGame.

**How To Install**

```bash
git clone https://github.com/ekorudiawan/gym-robot-arm.git
cd gym_imp
cd gym-robot-arm
pip install -e .
```

**Dependencies**
* OpenAI Gym
* PyGame
* Scipy
* Stable_baseline3

**Testing Environment**

```python
import gym 

env = gym.make('gym_robot_arm:robot-arm-v0')

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()


**Rebuild Training**
```bash
cd gym_imp
cd gym-robot-arm
pip install -e .
python3 train*.py // there are multiple training options
```
Then training information will be logged, you can use tensorboard to observe
```bash
tensorboard --logdir==
```

**render from trained model**
```python
  env = gym.make('gym_robot_arm:robot-arm-v1')
  env.seed(0)
  model_n = './td3_robot_model.zip' # can be replaced with other models
  model = TD3.load(model_n)
  print(model_n)
  obs = env.reset()
  ep_reward_list = []
  whether_done_list = []
  for i_episode in range(30):
      episode_reward = 0
      observation = env.reset()
      done = False
      for t in range(100):
          env.render()
          action, _states = model.predict(obs)
          # test
          print(action)
          obs, reward, done, info = env.step(action)
          episode_reward += reward
          if done:
              print("Episode finished after {} timesteps".format(t+1))
              print("Reward:", episode_reward)
              break
      ep_reward_list.append(episode_reward)
      whether_done_list.append(done)
  env.close()

  print(ep_reward_list)
  print(statistics.mean(ep_reward_list))
  print(whether_done_list)
  print(sum(whether_done_list)/len(whether_done_list))
```
