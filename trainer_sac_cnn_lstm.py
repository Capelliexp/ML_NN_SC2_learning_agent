from customized_environments.envs.my_agent import CustomAgent

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.policies import CnnLstmPolicy

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines import PPO2

from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])

# create vectorized environment
#env = gym.make('defeat-zerglings-banelings-v0')
#eng = CustomAgent()
env = DummyVecEnv([lambda: CustomAgent()])

# use ppo2 to learn and save the model when finished
#model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="gym_ouput/log/")
model = PPO2(
    CnnPolicy,
    env, 
    nminibatches = 1,
    verbose=1, 
    tensorboard_log="gym_ouput/log/"
    )


#model = PPO2.load("gym_ouput/NN")  # load existing network

for i in range(1,20):
    save_name = "gym_ouput/NN" + i.__str__()

    model.learn(total_timesteps=int(2e4), tb_log_name="run", reset_num_timesteps=False)
    model.save(save_name)
