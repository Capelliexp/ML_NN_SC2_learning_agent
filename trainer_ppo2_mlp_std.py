from customized_environments.envs.my_agent import CustomAgent

import gym

from stable_baselines.common.policies import MlpPolicy

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines import PPO2

from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])

name = "ppo2_mlp_std"
learn_type='PPO2'
start_value = 0

# create vectorized environment
env = DummyVecEnv([lambda: CustomAgent(learn_type=learn_type)])



model = PPO2(
    MlpPolicy,
    env, 
    learning_rate = 0.1,
    nminibatches = 8,
    verbose=1, 
    tensorboard_log="gym_ouput/" + name + "/log/"
    )


model.setup_model()

if start_value > 0:
    model.load("gym_ouput/" + name + "/it" + str(start_value), env=env)

i = 1
while True:
    save_name = "gym_ouput/" + name + "/it" + (i+start_value).__str__()

    model.learn(total_timesteps=int(3e3), tb_log_name="log", reset_num_timesteps=False)
    model.save(save_name)
    i += 1

