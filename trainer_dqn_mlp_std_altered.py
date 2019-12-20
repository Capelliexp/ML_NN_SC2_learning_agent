from customized_environments.envs.my_agent import CustomAgent

import gym

from stable_baselines.deepq.policies import MlpPolicy

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines import DQN

from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])

name = "dqn_mlp_std_altered"
learn_type='DQN'
start_value = 3

# create vectorized environment
env = DummyVecEnv([lambda: CustomAgent(learn_type=learn_type)])



model = DQN(
    MlpPolicy,
    env, 
    learning_rate = 0.4,
    exploration_fraction = 0.4,
    train_freq = 1,
    double_q = True,
    learning_starts = 1920,
    target_network_update_freq = 3840,
    param_noise = False,
    verbose=1, 
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

    model.learn(total_timesteps=int(5e4), tb_log_name="log", reset_num_timesteps=False)
    model.save(save_name)
    i += 1

