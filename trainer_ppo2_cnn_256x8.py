from customized_environments.envs.my_agent import CustomAgent

import gym

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.policies import CnnLstmPolicy

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines import PPO2
from stable_baselines import SAC

from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])

name = "256x8"

# create vectorized environment
env = DummyVecEnv([lambda: CustomAgent()])

policy_kwargs = dict(net_arch=[256, 256, 256, 256, 256, 256, 256, 256])

model = PPO2(
    CnnPolicy,
    env, 
    learning_rate = 0.1,
    nminibatches = 8,
    verbose=1, 
    policy_kwargs=policy_kwargs,
    tensorboard_log="gym_ouput/" + name + "/log/"
    )

model.setup_model()

start_value = 0
if start_value > 0:
    model.load("gym_ouput/" + name + "/it" + str(start_value), env=env)

i = 1
while True:
    save_name = "gym_ouput/" + name + "/it" + (i+start_value).__str__()

    model.learn(total_timesteps=int(1e4), tb_log_name="log", reset_num_timesteps=False)
    model.save(save_name)
    i += 1

