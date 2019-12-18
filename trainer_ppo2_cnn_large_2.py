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
env = DummyVecEnv([lambda: CustomAgent()])

policy_kwargs = dict(net_arch=[128, 64, 64, 16])

model = PPO2(
    CnnPolicy,
    env, 
    nminibatches = 1,
    verbose=1, 
    policy_kwargs=policy_kwargs,
    tensorboard_log="gym_ouput/PPO2_CNN_large_2/log/"
    )

model.setup_model()

for i in range(1,20):
    save_name = "gym_ouput/PPO2_CNN_large_2/it" + i.__str__()
    #save_name = "gym_ouput/PPO2_CNN/model"

    model.learn(total_timesteps=int(1e4), tb_log_name="log", reset_num_timesteps=False)
    model.save(save_name)

