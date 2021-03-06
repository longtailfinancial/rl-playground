

import os
from numpy.core.fromnumeric import mean
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

import random
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from gym import spaces, Env
import numpy as np

class StakingEnv(Env):
  """
  Custom Environment that follows gym interface.
  """
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['console']}
  # Define constants for clearer code and rewards
  DO_NOTHING = 0
  STAKE = 1
  UNSTAKE = 2

  def __init__(self, ep_length: int = 20, balance: int = 30):
    super(StakingEnv, self).__init__()

    # self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(5), spaces.Discrete(100)))
    self.action_space = spaces.MultiDiscrete([3, 5, 4])
    self.observation_space = spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32)
    self.ep_length = ep_length
    self.current_step = 0
    self.current_action = self.action_space.sample()
    self.num_resets = -1  # Becomes 0 after __init__ exits.
    self.balance = balance
    self.performance = []
    self.reset()

  def reset(self):
      self.current_step = 0
      self.balance = 100
      self.num_resets += 1
      self.accumulated_reward = 0
      self.state = [10.0, 10.0, 10.0, 10.0, 10.0]
      self.gradient = [0.0, 0.0, 0.0, 0.0, 0.0]
      return np.array(self.state).astype(np.float32)

  def step(self, action):
      reward = self._get_reward(action)
      self._get_next_state(action)
      self.current_step += 1
      self.accumulated_reward += reward
      if self.current_step >= self.ep_length or self.balance <= 0.0:
        done = True
      else:
        done = False
      # self.state = map(np.float32, self.state)
      return np.array(self.state).astype(np.float32), reward, done, {}

  def _get_next_state(self, action):
      self._define_state(action)
      # randomly stake in 30% of the time
      self._random_action()

  def _define_state(self, action):
      prev_state = self.state.copy()

      stake = 10 * action[2]/4

      if action[0] == StakingEnv.DO_NOTHING:
        return
      elif action[0] == StakingEnv.STAKE:
        self.state[action[1]] += stake
        self.balance -= stake
      elif action[0] == StakingEnv.UNSTAKE:
        self.state[action[1]] -= stake
        self.balance += stake
      else:
        return
      # calculate gradient
      numerator = self.state[action[1]] - prev_state[action[1]]
      denominator = prev_state[action[1]]
      if denominator == 0.0:
        denominator = 1
      self.gradient[action[1]] = numerator/denominator

  def _random_action(self):
      r = random.random()
      random_staker = 0
      if self.current_step % 3 == 0:
        random_staker = 10 * r
      else:
        random_staker = -10 * r
      rs = random.choice(range(5))
      # print(f'Random pool: {rs} Random stake: {random_staker}'
      self.state[rs] += random_staker

  def _get_reward(self, action):
      self.current_action = action
      reward = 0
      if action[0] == StakingEnv.DO_NOTHING:
        reward = 0
      elif action[0] == StakingEnv.STAKE:
        reward = 1 * self.gradient[action[1]]
      elif action[0] == StakingEnv.UNSTAKE:
        reward = -1 * self.gradient[action[1]]
      else:
        pass
      if self.balance < 20.0:
        reward -= 1
      return reward

  def render(self, mode='console'):
      print("Step {}".format(self.current_step))
      print("Action: ", self.current_action)
      print("Reward: ", self.accumulated_reward)
      print("State: ", self.state)
      print("Gradient: ", self.gradient)
      print("Balance: ", self.balance)
      self.performance.append([
        self.current_step,
        self.current_action,
        self.accumulated_reward,
        self.state,
        self.gradient,
        self.balance,
      ])

    
env = StakingEnv()
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)

env = StakingEnv(10)
# # multiprocess environment
# env = make_vec_env(env, n_envs=4)

obs = env.reset()
env.render()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

# n_steps = 90
# for step in range(n_steps):
#   print("Step {}".format(step + 1))
#   action = env.action_space.sample()
#   print("Action {}".format(action))
#   print(f'Gradient= {env.gradient}')
#   print(f'Balance= {env.balance}')
#   obs, reward, done, info = env.step(action)
#   print('obs=', obs, 'reward=', reward, 'done=', done,'\n')
#   # env.render()
#   if done:
#     print("Episode end reached!", "reward=", reward)
#     break

def make_env(env, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# num_cpu = 4  # Number of processes to use
# Create the vectorized environment
# env = DummyVecEnv([make_env(StakingEnv(100), i) for i in range(num_cpu)])

# # Instantiate the agent
# model = PPO(MlpPolicy, env, verbose=1)
# # Train the agent
# train_steps=25000
# train_steps=2500
# model.learn(total_timesteps=int(train_steps))
# # Save the agent
# model.save("ppo_stake")
# del model  # delete trained model to demonstrate loading

# Load the trained agent
model = PPO.load("ppo_stake", env)

# print(model)
# # Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# print(f'mean_reward: {mean_reward}, std_reward: {std_reward}')

obs = env.reset()
for i in range(20):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    
performance = pd.DataFrame(env.performance, columns=['Step','Action','Reward','State', 'Gradient', 'Balance'])
performance.to_csv('agent_performance.csv')
