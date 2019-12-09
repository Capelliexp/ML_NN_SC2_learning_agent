import gym
from customized_environments.envs.my_agent import CustomAgent
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])

# create vectorized environment
env = gym.make('defeat-zerglings-banelings-v0')
eng = CustomAgent()
env = DummyVecEnv([lambda: CustomAgent()])

# use ppo2 to learn and save the model when finished
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="log/")
model.learn(total_timesteps=int(1e5), tb_log_name="fisrt_rum", reset_num_timesteps=False)
model.save("model/dbz_ppo")