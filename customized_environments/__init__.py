from gym.envs.registration import register

register(
    id='defeat-zerglings-banelings-v0',
    entry_point='customized_environments.envs:CustomAgent',
)
