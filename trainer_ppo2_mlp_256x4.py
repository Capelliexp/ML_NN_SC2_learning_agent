from customized_environments.envs.my_agent import CustomAgent

import gym

from stable_baselines.common.policies import MlpPolicy

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines import PPO2

from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])

name = "ppo2_mlp_256x4"
learn_type='PPO2'
start_value = 16

# create vectorized environment
env = DummyVecEnv([lambda: CustomAgent(learn_type=learn_type)])

policy_kwargs = dict(net_arch=[256, 256, 256, 256])

model = PPO2(
    MlpPolicy,
    env, 
    learning_rate = 0.1,
    nminibatches = 8,
    verbose=1, 
    policy_kwargs=policy_kwargs,
    tensorboard_log="gym_ouput/" + name + "/log/"
    )






model.setup_model()

if start_value > 0:
    try:
        model.load("gym_ouput/" + name + "/it" + str(start_value + 1), env=env)
        print("\n\nOBS! this is not the latest NN load point\n\n")
    except:
        try:
            model.load("gym_ouput/" + name + "/it" + str(start_value), env=env)
        except:
            print("\n\nOBS! invalid load point\n\n")

i = 1
while True:
    save_name = "gym_ouput/" + name + "/it" + (i+start_value).__str__()

    model.learn(total_timesteps=int(1e4), tb_log_name="log", reset_num_timesteps=False)
    model.save(save_name)
    i += 1

