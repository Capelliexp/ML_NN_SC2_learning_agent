from customized_environments.envs.my_agent import CustomAgent

import gym

from stable_baselines.deepq.policies import CnnPolicy

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines import DQN

from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])

name = "dqn_cnn_std"
learn_type='DQN'
start_value = 0

# create vectorized environment
env = DummyVecEnv([lambda: CustomAgent(learn_type=learn_type)])



model = DQN(
    CnnPolicy,
    env, 
    learning_rate = 0.1,
    double_q = True,
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

    model.learn(total_timesteps=int(3e3), tb_log_name="log", reset_num_timesteps=False)
    model.save(save_name)
    i += 1

