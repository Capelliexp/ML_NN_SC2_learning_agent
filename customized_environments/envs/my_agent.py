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
        'map_name': "DefeatZerglingsAndBanelings",
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
        'step_mul': 8,
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

        """
        self.observation_space = spaces.Box(    #preliminary
            low=0,
            high=64,
            shape=(19,3),
            dtype=np.uint8  #new
        )
        """

        """
        self.action_space = spaces.Discrete(123)  #preliminary
        """

        """
        self.observation_space = spaces.Dict({
            'tensor_ints' : spaces.Dict({
                'HP' : spaces.Discrete(1),
                'allies' : spaces.Discrete(1),
                'enemies' : spaces.Discrete(1),
                'pos' : spaces.Box(low = 0, high = 64, shape = (2,1), dtype = np.int32),
                'goal' : spaces.Box(low = 0, high = 64, shape = (2,1), dtype = np.int32)
            }),
            'tensor_floats' : spaces.Dict({
                'attack_reset' : spaces.Box(low = 0, high = 3, shape = (1,1), dtype = np.float32),
                'distance_to_goal' : spaces.Box(low = 0, high = 100, shape = (1,1), dtype = np.float32)
            }),
            'textures' : spaces.Dict({
                'player_relative' : spaces.Box(low = 0, high = 255, shape = (64,64), dtype = np.float32),
                'selected' : spaces.Box(low = 0, high = 255, shape = (64,64), dtype = np.float32)
            })
        })

        self.action_space = spaces.Dict({
            'tensor_movement' : spaces.Dict({
                'x' : spaces.Box(low = -1, high = 1, shape = (1,1), dtype = np.float32),
                'y' : spaces.Box(low = -1, high = 1, shape = (1,1), dtype = np.float32),
            }),
            'tensor_actions' : spaces.Dict({
                'attack_closest' : spaces.Discrete(1)
            })
        })
        """

        """
        self.observation_space = spaces.Dict({
            'HP' : spaces.Discrete(1),
            'allies' : spaces.Discrete(1),
            'enemies' : spaces.Discrete(1),
            'pos' : spaces.Box(low = 0, high = 64, shape = (2,1), dtype = np.int32),
            'goal' : spaces.Box(low = 0, high = 64, shape = (2,1), dtype = np.int32),
            'attack_reset' : spaces.Box(low = 0, high = 3, shape = (1,1), dtype = np.float32),
            'distance_to_goal' : spaces.Box(low = 0, high = 100, shape = (1,1), dtype = np.float32),
            'player_relative' : spaces.Box(low = 0, high = 255, shape = (64,64), dtype = np.float32),
            'selected' : spaces.Box(low = 0, high = 255, shape = (64,64), dtype = np.float32)
        })

        self.action_space = spaces.Dict({
            'x' : spaces.Box(low = -1, high = 1, shape = (1,1), dtype = np.float32),
            'y' : spaces.Box(low = -1, high = 1, shape = (1,1), dtype = np.float32),
            'attack_closest' : spaces.Discrete(1)
        })
        """

        """
        self.observation_space = spaces.Tuple((
            spaces.Discrete(1), # HP
            spaces.Discrete(1), # allies
            spaces.Discrete(1), # enemies
            spaces.Box(low = 0, high = 64, shape = (2,1), dtype = np.int32),    # pos
            spaces.Box(low = 0, high = 64, shape = (2,1), dtype = np.int32),    # goal
            
            spaces.Box(low = 0, high = 3, shape = (1,1), dtype = np.float32),   # attack_reset
            spaces.Box(low = 0, high = 100, shape = (1,1), dtype = np.float32), # distance_to_goal
            
            spaces.Box(low = 0, high = 255, shape = (64,64), dtype = np.float32),   # player_relative
            spaces.Box(low = 0, high = 255, shape = (64,64), dtype = np.float32)    # selected
        ))
        """

        """
        self.action_space = spaces.Tuple((
            spaces.Box(low = -1, high = 1, shape = (1,1), dtype = np.float32),  # x
            spaces.Box(low = -1, high = 1, shape = (1,1), dtype = np.float32),  # y
            spaces.Discrete(1)  # attack_closest
        ))
        """

        #self.observation_space = spaces.Box(low = 0, high = 255, shape = (64*2+1, 64, 3), dtype = np.float32)
        #self.observation_space = spaces.Box(low = 0, high = 255, shape = (64*2, 64, 3), dtype = np.float32)
        self.observation_space = spaces.Box(low = 0, high = 255, shape = (64*2, 64), dtype = np.float32)

        #self.action_space = spaces.Discrete(3)
        #self.action_space = spaces.Box(low = -1, high = 1, shape = (3, 3), dtype = np.float32)
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

        raw_obs = self.env.reset()[0]
        
        return self.get_derived_obs(raw_obs)

    def get_derived_obs(self, raw_obs):
        # manualy convert raw_obs from pysc2 to the user defined input obs
        # return array with observation containing self.observation_space variables

        #obs = np.zeros((64*2+1, 64, 3), dtype = np.float32)
        #obs = np.zeros((19,3), dtype=np.uint8)

        marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)   # team 1: my team
        zerglings = self.get_units_by_type(raw_obs, units.Zerg.Zergling, 4) # team 4: enemy team
        banelings = self.get_units_by_type(raw_obs, units.Zerg.Baneling, 4)

        marine_it = self.steps%len(marines)

        goal_dist = math.sqrt((marines[marine_it].x - self.goal[0])**2 + (marines[marine_it].y - self.goal[1])**2)

        #obs = raw_obs.observation["feature_screen"][features.SCREEN_FEATURES.player_relative.index]
        #obs = np.tile(obs, raw_obs.observation["feature_screen"][features.SCREEN_FEATURES.selected.index])

        #obs = raw_obs.observation["feature_screen"][features.SCREEN_FEATURES.player_relative.index]
        #obs = obs + raw_obs.observation["feature_screen"][features.SCREEN_FEATURES.selected.index]

        obs = np.concatenate((
            raw_obs.observation["feature_screen"][features.SCREEN_FEATURES.player_relative.index],
            raw_obs.observation["feature_screen"][features.SCREEN_FEATURES.selected.index]
            ))

        """obs = np.tile(obs, [
            marines[marine_it].health,          # HP
            len(marines),                       # allies
            len(zerglings) + len(banelings),    # enemies
            [marines[marine_it].x, marines[marine_it].y],   # pos
            self.goal,                          # goal
            marines[marine_it].weapon_cooldown, # attack_reset
            goal_dist                           # distance_to_goal
        ])"""
        #obs = np.tile(obs, np.zeros((1, 64-9, 3), dtype = np.float32))
        """obs = obs + [
            marines[marine_it].health,          # HP
            len(marines),                       # allies
            len(zerglings) + len(banelings),    # enemies
            [marines[marine_it].x, marines[marine_it].y],   # pos
            self.goal,                          # goal
            marines[marine_it].weapon_cooldown, # attack_reset
            goal_dist                           # distance_to_goal
        ]"""
        #obs = obs + np.zeros((1, 64-9, 3), dtype = np.float32)

        """
        obs = self.observation_space
        obs['tensor_ints']['HP'] = marines[marine_it].health
        obs['tensor_ints']['allies'] = len(marines)
        obs['tensor_ints']['enemies'] = len(zerglings) + len(banelings)
        obs['tensor_ints']['pos'] = marines[marine_it].position
        obs['tensor_ints']['goal'] = self.goal

        obs['tensor_floats']['attack_reset'] = marines[marine_it].weapons[0].cooldown
        obs['tensor_floats']['distance_to_goal'] = math.sqrt(
            (marines[marine_it].position.x - self.goal[0])**2 + (marines[marine_it].position.y - self.goal[1])**2)

        obs['textures']['player_relative'] = raw_obs.observation["feature_screen"][features.SCREEN_FEATURES.player_relative.index]
        obs['textures']['selected'] = raw_obs.observation["feature_screen"][features.SCREEN_FEATURES.selected.index]
        """

        #player_relative = raw_obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index]
        #player_relative = raw_obs.observation["feature_screen"][features.SCREEN_FEATURES.player_relative.index]

        #if player_relative is not None:
        #    print("YAS!")
        

        #self.marines = []
        #self.banelings = []
        #self.zerglings = []

        #for i, m in enumerate(marines):
        #    self.marines.append(m)
        #    obs[i] = np.array([m.x, m.y, m[2]])

        #for i, b in enumerate(banelings):
        #    self.banelings.append(b)
        #    obs[i+9] = np.array([b.x, b.y, b[2]])

        #for i, z in enumerate(zerglings):
        #    self.zerglings.append(z)
        #    obs[i+13] = np.array([z.x, z.y, z[2]])
        
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
        if action == 0:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
        elif action <= 32:
            derived_action = np.floor((action-1)/8)
            idx = (action-1)%8
            if derived_action == 0:
                action_mapped = self.move(idx, 0, -2)
            elif derived_action == 1:
                action_mapped = self.move(idx, 0, 2)
            elif derived_action == 2:
                action_mapped = self.move(idx, -2, 0)
            else:
                action_mapped = self.move(idx, 2, 0)
        else:
            eidx = np.floor((action-33)/9)
            aidx = (action-33)%9
            action_mapped = self.attack(aidx, eidx)
        
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