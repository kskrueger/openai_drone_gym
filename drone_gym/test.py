import cv2
import gym


#import drone_gym.env
import drone_gym
import drone_gym.envs
import numpy as np
import argparse
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

env = gym.make('airsim-drone-v0')
env._max_episode_steps = 1

model = DQN(MlpPolicy, env,  buffer_size=2500, verbose=1)
model.learn(total_timesteps=100, log_interval=4)

model = DQN.load("dqn_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    print("Predict action: ", action)
    obs, reward, done, info = env.step(action)
    print("Reward: ", reward)
    print("Obs: ", obs)
    # cv2.waitKey(1)
    # env.render()
    model.save("dqn_pendulum")
    if done:
      obs = env.reset()

