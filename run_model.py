from customized_environments.envs.my_agent import CustomAgent

import gym

from stable_baselines.deepq.policies import CnnPolicy

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines import PPO2
from stable_baselines import DQN

from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])

name = "dqn_mlp_std_altered"
learn_type='DQN'
model_iteration = 34

# create vectorized environment
env = DummyVecEnv([lambda: CustomAgent(learn_type=learn_type)])

if model_iteration > 0:
    if learn_type == "DQN":
        model = DQN.load("gym_ouput/" + name + "/it" + str(model_iteration), env=env)
    elif learn_type == "PPO2":
        model = PPO2.load("gym_ouput/" + name + "/it" + str(model_iteration), env=env)
else:
    print("invalid model_iteration")
    exit

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    

