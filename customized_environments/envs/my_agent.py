"""gym"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding

"""SC2"""
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env

"""other"""
import logging
import math
import numpy as np

class CustomAgent(gym.Env):
    metadata = {'render.modes':['human']}

    default_settings = {    #OBS! fix
        '_only_use_kwargs': None,
        'map_name': "simple_1v1",
        'battle_net_map': False,
        'players': [sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.hard)],
        'agent_interface_format': features.AgentInterfaceFormat(
            #feature_dimensions = None,
            feature_dimensions = features.Dimensions(screen=64, minimap=64),
            rgb_dimensions = None,
            raw_resolution = 64,
            action_space = actions.ActionSpace.RAW,
            camera_width_world_units = None,
            #camera_width_world_units = 64,
            #use_feature_units = False,
            use_feature_units = True,
            use_raw_units = True,
            #use_raw_actions = False,
            use_raw_actions = True,
            max_raw_actions = 512,
            max_selected_units = 30,
            #max_selected_units = 1,
            use_unit_counts = False,
            use_camera_position = False,
            #use_camera_position = True,
            show_cloaked = False,
            show_burrowed_shadows = False,
            show_placeholders = False,
            hide_specific_actions = True,
            action_delay_fn = None,
            send_observation_proto = False,
            crop_to_playable_area = False,
            raw_crop_to_playable_area = False,
            allow_cheating_layers = False,
            add_cargo_to_units = False
            ),
        'discount': 1,
        'discount_zero_after_timeout': False,
        'visualize': False,
        #'visualize': True,
        #'step_mul': None,
        'step_mul': 4,
        'realtime': False,    #should be false during training
        'save_replay_episodes': 0,
        'replay_dir': None,
        'replay_prefix': None,
        'game_steps_per_episode': None,
        'score_index': None,
        'score_multiplier': None,
        'random_seed': None,
        'disable_fog': False,
        'ensure_available_actions': True,
        'version': None
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.env = None

        self.marines = []
        self.banelings = []
        self.zerglings = []

        self.goal = [0,0]

        self.observation_space = spaces.Box(low = 0, high = 255, shape = (64*2, 64, 1), dtype = np.float32)

        self.action_space = spaces.Box(np.array([-1, -1, 0]), np.array([1, 1, 1]))

        self.episodes = 0
        self.steps = 0

    def reset(self):
        self.episodes += 1
        self.steps = 0

        self.goal = [0,0]
        
        if self.env is None:
            args = {**self.default_settings, **self.kwargs}
            self.env =  sc2_env.SC2Env(**args)
        
        self.marines = []
        self.banelings = []
        self.zerglings = []

        self.current_marine_it = 0

        raw_obs = self.env.reset()[0]
        
        return self.get_derived_obs(raw_obs)

    def get_derived_obs(self, raw_obs):
        # manualy convert raw_obs from pysc2 to the user defined input obs
        # return array with observation containing self.observation_space variables

        self.marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)   # team 1: my team
        self.zerglings = self.get_units_by_type(raw_obs, units.Zerg.Zergling, 4) # team 4: enemy team
        self.banelings = self.get_units_by_type(raw_obs, units.Zerg.Baneling, 4)

        if len(self.marines) > 0:
            self.current_marine_it = self.steps%len(self.marines)
        else:
            self.current_marine_it = 0

        obs = np.concatenate((
            raw_obs.observation["feature_screen"][features.SCREEN_FEATURES.player_relative.index],
            raw_obs.observation["feature_screen"][features.SCREEN_FEATURES.selected.index]
            ))

        obs.resize((64*2, 64, 1))
        
        return obs

    def get_units_by_type(self, obs, unit_type, player_relative=0):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type and unit.alliance == player_relative
            ]

    def step(self, action):
        self.steps += 1

        raw_obs = self.take_action(action)
        reward = raw_obs.reward
        obs = self.get_derived_obs(raw_obs)

        return obs, reward, raw_obs.last(), {}  #new

    def take_action(self, action):
        # map value to an action

        selected = self.marines[self.current_marine_it]

        x = 1 if action[0] > 0 else -1
        y = 1 if action[1] > 0 else -1
        attack = action[2]

        if attack > 0.5:
            action_mapped = actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, [selected.x, selected.y])
        else:
            new_pos = [selected.x + x*2, selected.y + y*2]
            action_mapped = actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)

        raw_obs = self.env.step([action_mapped])[0]

        return raw_obs

    def move(self, idx, diff_x=0, diff_y=0):
        try:
            selected = self.marines[idx]
            new_pos = [selected.x+diff_x, selected.y+diff_y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()
        
    def attack(self, aidx, eidx):
        try:
            selected = self.marines[aidx]
            if edix>3:
                target = self.zerglines[eidx-4]
            else:
                target = self.banelings[eidx]
            return actions.RAW_FUNCTIONS.Attack_unit("now", selected.tag, target.tag) #OBS! used to be "targeted"
        except:
            return actions.RAW_FUNCTIONS.no_op()

    

    def render(self, mode='human', close=False):
        pass

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()