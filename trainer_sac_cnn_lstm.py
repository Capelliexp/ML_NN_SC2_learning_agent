from customized_environments.envs.my_agent import CustomAgent

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.policies import CnnLstmPolicy

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines import PPO2
from stable_baselines import SAC

from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])

# create vectorized environment
#env = gym.make('defeat-zerglings-banelings-v0')
#eng = CustomAgent()
env = DummyVecEnv([lambda: CustomAgent()])

# use ppo2 to learn and save the model when finished
#model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="gym_ouput/log/")
model = SAC(
    CnnPolicy,
    env, 
    verbose=1, 
    tensorboard_log="gym_ouput/log/"
    )


#model = PPO2.load("gym_ouput/NN")  # load existing network

for i in range(1,20):
    save_name = "gym_ouput/SAC_CNN_LSTM/it" + i.__str__()

    model.learn(total_timesteps=int(3e4), tb_log_name="SAC_CNN_LSTM", reset_num_timesteps=False)
    model.save(save_name)

