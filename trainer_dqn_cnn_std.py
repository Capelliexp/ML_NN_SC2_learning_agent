from customized_environments.envs.my_agent import CustomAgent

import gym

from stable_baselines.deepq.policies import CnnPolicy

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines import PPO2
from stable_baselines import SAC
from stable_baselines import DQN

from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])

name = "dqn_std"

# create vectorized environment
env = DummyVecEnv([lambda: CustomAgent(learn_type='DQN')])



model = DQN(
    CnnPolicy,
    env, 
    learning_rate = 0.1,
    double_q = True,
    verbose=1, 
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

