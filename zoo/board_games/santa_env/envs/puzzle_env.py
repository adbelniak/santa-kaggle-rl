import copy
from typing import List

import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd
import ast
from sympy.combinatorics import Permutation
from ding.envs import BaseEnv
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from ding.envs import BaseEnv, BaseEnvTimestep


# @ENV_REGISTRY.register('santa_env')
class SantaEnv(gymnasium.Env):
    
    config = dict(
        # (str) The name of the environment registered in the environment registry.
        env_name="santa_env",
        # (str) The render mode. Options are 'None', 'state_realtime_mode', 'image_realtime_mode' or 'image_savefile_mode'.
        # If None, then the game will not be rendered.
        render_mode=None,
        # (str) The format in which to save the replay. 'gif' is a popular choice.
        replay_format='gif',
        # (str) A suffix for the replay file name to distinguish it from other files.
        replay_name_suffix='eval',
        # (str or None) The directory in which to save the replay file. If None, the file is saved in the current directory.
        replay_path=None,
        # (bool) Whether to scale the actions. If True, actions are divided by the action space size.
        act_scale=True,
        # (bool) Whether to use the 'channel last' format for the observation space.
        # If False, 'channel first' format is used.
        channel_last=True,
        # (str) The type of observation to use. Options are 'raw_board', 'raw_encoded_board', and 'dict_encoded_board'.
        obs_type='dict_encoded_board',
        # (bool) Whether to normalize rewards. If True, rewards are divided by the maximum possible reward.
        reward_normalize=False,
        # (float) The factor to scale rewards by when reward normalization is used.
        reward_norm_scale=100,
        # (str) The type of reward to use. 'raw' means the raw game score. 'merged_tiles_plus_log_max_tile_num' is an alternative.
        reward_type='raw',
        # (int) The maximum tile number in the game. A game is won when this tile appears. 2**11=2048, 2**16=65536
        max_tile=int(1000),
        # (int) The number of steps to delay rewards by. If > 0, the agent only receives a reward every this many steps.
        delay_reward_step=0,
        # (float) The probability that a random agent is used instead of the learning agent.
        prob_random_agent=0.,
        # (int) The maximum number of steps in an episode.
        max_episode_steps=int(1e6),
        # (bool) Whether to collect data during the game.
        is_collect=True,
        # (bool) Whether to ignore legal actions. If True, the agent can take any action, even if it's not legal.
        ignore_legal_actions=True,
        # (bool) Whether to flatten the observation space. If True, the observation space is a 1D array instead of a 2D grid.
        need_flatten=False,
        # (int) The number of possible tiles that can appear after each move.
        num_of_possible_chance_tile=2,
        # (numpy array) The possible tiles that can appear after each move.
        possible_tiles=np.array([2, 4]),
        # (numpy array) The probabilities corresponding to each possible tile.
        tile_probabilities=np.array([0.9, 0.1]),
    )
     
    def __init__(self,cfg: dict,max_step=20,render_mode='rgb_array'):
        super().__init__()
        self._cfg = cfg
        self.render_mode = render_mode
        self.observation_space = spaces.MultiDiscrete(np.ones(1)*(24 * 26))
        self.action_space = spaces.Discrete(6)
        self.reward_range = (-np.inf,1)
        self.puzzle_info = pd.read_csv('puzzle_info.csv')
        self.puzzles = pd.read_csv('puzzles.csv')
        self.possible_action = None
        self.metadata = {'render.modes': ['human']}
        self.current_puzzle = None
        self.current_state = None
        self.solution_state = None
        self.nb_move_played = 0
        self.num_wildcards = 0
        self.penality_coeff = 0.1
        self.current_puzzle_type = None
        self.current_puzzle_id = None
        self.puzzle_types = set()
        self.dict_letter_to_label = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,'F': 5,
                                'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10,'L': 11,
                                'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16,'R': 17,
                                'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,'X': 23,
                                'Y': 24, 'Z': 25}

        for i in (range(self.puzzles.shape[0])):
            self.puzzle_types.add(self.puzzles.iloc[i, 1])
        self.puzzle_types = list(self.puzzle_types)
        self.max_step = max_step
        self.reward_space = gymnasium.spaces.Box(
                low=-10, high=10, shape=(1,), dtype=np.float32
            )

        
    def get_allowed_move(self, puzzle_type,puzzle_info):
        """
        Gets the allowed moves for a given puzzle type in the Santa 2023 Kaggle competition.

        Parameters:
        puzzle_type (str): The type of the puzzle.
        puzzle_info (pd.DataFrame): The puzzle info dataframe.

        Returns:
        dict: A dictionary of allowed moves. The keys are the moves and the values are the new positions of the colors in the puzzle after the move.
        """
        allowed_moves = {}
        for i in range(puzzle_info.shape[0]):
            if puzzle_info.iloc[i].puzzle_type == puzzle_type:
                allowed_moves = ast.literal_eval(puzzle_info.iloc[i].allowed_moves)
                break
        return allowed_moves

    def choose_move_among_allowed(self,allowed_moves,action_index):
        """
        Chooses a move among the allowed moves for a given action index in the Santa 2023 Kaggle competition.

        Parameters:
        allowed_moves (dict): A dictionary of allowed moves. The keys are the moves and the values are the new positions of the colors in the puzzle after the move.
        action_index (int): The index of the action.

        Returns:
        list: The move to be performed. It is a list of integers representing the new positions of the colors in the puzzle after the move.
        """

        return allowed_moves[str(list(allowed_moves.keys())[action_index])]
        
    def is_solution(self, state1, state2, num_wildcards=0):
        """
        Checks if a given state is a solution state in the Santa 2023 Kaggle competition.

        Parameters:
        state1 (str): The state to be checked. It is a string of characters separated by semicolons (;). Each character represents a color on the puzzle.
        state2 (str): The solution state. It is a string of characters separated by semicolons (;). Each character represents a color on the puzzle.
        num_wildcards (int, optional): The number of wildcards allowed. Defaults to 0.

        Returns:
        bool: True if the given state is a solution state, False otherwise.
        """
        return  state1 == state2
        
    def reward_function(self, current_state,solution_state,nb_move_played,num_wildcards=0,
                    penality_coeff=0.1):
        """
        Calculates the reward for a given move in the Santa 2023 Kaggle competition.

        Parameters:
        current_state (str): The current state of the puzzle. It is a string of characters separated by semicolons (;). Each character represents a color on the puzzle.
        solution_state (str): The solution state of the puzzle. It is a string of characters separated by semicolons (;). Each character represents a color on the puzzle.
        nb_move_played (int): The number of moves played so far.
        num_wildcards (int, optional): The number of wildcards allowed. Defaults to 0.
        penality_coeff (float, optional): The penality coefficient. Defaults to 0.1.

        Returns:
        float: The reward for the given move.
        """

        reward = 0
        for i in range(len(current_state)):
            if current_state[i] == solution_state[i]:
                reward -= 0.01
            elif current_state[i] in solution_state:
                reward -= 0.04
            else:
                reward -= -0.0005
        if self.is_solution(current_state, solution_state, num_wildcards):
            return 1
        else:

            return  reward
        
    def _encode_state(self, obs):
        enc_obs = []
        for o in obs:
            vec = np.zeros(len(self.dict_letter_to_label.keys()))
            vec[self.dict_letter_to_label[o]] = 1
            enc_obs.append(vec)
        enc_obs = np.asarray(enc_obs)
        return np.expand_dims(enc_obs, axis=0)
    
    def play_move(self, state, move):
        """
        Simulates a move in the Santa 2023 Kaggle competition.

        Parameters:
        state (str): The initial state of the puzzle. It is a string of characters separated by semicolons (;). Each character represents a color on the puzzle.
        move (list): The move to be performed. It is a list of integers representing the new positions of the colors in the puzzle after the move.

        Returns:
        str: The state of the puzzle after the move. It is a string of characters separated by semicolons (;). Each character represents a color on the puzzle.
        """

        new_state = state.copy()
        move_p = Permutation(move)
        return move_p(new_state)
        
    def reset(self,seed=None,options=None):
        # get a random puzzle
        #self.current_puzzle = self.puzzles.sample()
        # get puzzle 0 as a dataframe (like if I did .samples())
        super().reset(seed=seed)
        # here I get the first puzzle of the dataframe
        self.current_puzzle = self.puzzles.iloc[1, :]
        self.current_puzzle_type = self.current_puzzle.puzzle_type
        self.current_puzzle_id = self.current_puzzle.id
        self.current_state = self.current_puzzle.initial_state.split(";")
        # remove the first and last character
#         self.current_state = self.current_state[2:-2]
        self.solution_state = self.current_puzzle.solution_state.split(";")
        # remove the first and last character
#         self.solution_state = self.solution_state[2:-2]
        self.number_step_current_puzzle = 0
        self.possible_action = self.get_allowed_move(self.current_puzzle_type, self.puzzle_info)
        self.nb_move_played = 0
        self.num_wildcards = 0
        self._final_eval_reward = 0.0
        obs = {'observation':  self._encode_state(self.current_state), 'action_mask': self.get_mask(), 'to_play': -1}

        return obs

    def step(self,action):
        # play the move
        
        self.current_state = self.play_move((self.current_state), list(self.choose_move_among_allowed(self.possible_action, action)))
        self.number_step_current_puzzle += 1
        # check if the puzzle is solved
        done = self.is_solution(self.current_state, self.solution_state, self.num_wildcards)

        done = done or self.number_step_current_puzzle > self.max_step
        info = {}
        if done:
            reward = self.reward_function(self.current_state, self.solution_state, self.nb_move_played, self.num_wildcards, self.penality_coeff)
        else:
            reward = 0
        # update the number of moves played
        self._final_eval_reward += reward
        if done:

            if self.number_step_current_puzzle > self.max_step:

                print("max step reached")
            else:

                print("solved")
                print(f"target : {self.solution_state}")
                print(f"current state : {self.current_state}")
                """# plot the solution
                self.render(keep_plot=True)"""

            self.number_step_current_puzzle = 0
            info['eval_episode_return'] = self._final_eval_reward
        # calculate the reward
       

        self.nb_move_played += 1
        
        obs = {'observation':  self._encode_state(self.current_state), 'action_mask': self.get_mask(), 'to_play': -1}

        return BaseEnvTimestep(obs, reward, done, info)

    def get_number_of_moves_possible(self):
        return len(self.possible_action)


    # def close(self):
    #     super().close()
    #     plt.close()

    def get_info(self):
        return {'puzzle_type': self.current_puzzle_type,
                'puzzle_id': self.current_puzzle_id,
                'solution_state': self.solution_state,
                'nb_move_played': self.nb_move_played,
                'num_wildcards': self.num_wildcards,
                'penality_coeff': self.penality_coeff,
                'current_state': self.current_state}

    def list_of_letter_to_label(self,list_of_letter):
        # get the str as a list and remove the ;
        list_of_letter = list_of_letter
        dict_letter_to_label = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,'F': 5,
                                'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10,'L': 11,
                                'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16,'R': 17,
                                'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,'X': 23,
                                'Y': 24, 'Z': 25}
        # normalize the letter
        state_number = [dict_letter_to_label[letter] for letter in list_of_letter]
        # Example of normalization between 0 and 1
        normalized_state = [(num - (0)) / ((25) - (0)) for num in state_number]
        # pad with -1 to get the max_state_size
        normalized_state = normalized_state + [-1] * (self.max_state_size - len(normalized_state))

        return np.array(normalized_state, dtype=np.float32)

    def get_mask(self):
        """
        Get the mask of the possible actions. Put 0 for the impossible actions and 1 for the possible actions
        :return: mask
        """
        # number_of_possible_action = len(list(self.possible_action.keys()))
        # mask = np.zeros(self.action_space.n)
        # #mask = mask - np.inf

        # for i in range(number_of_possible_action):
        #     if i not in list(self.possible_action.keys()):
        #         mask[i] = 1


        return np.ones(self.action_space.n, 'int8')
    
    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        # when in collect phase, sometimes we need to normalize the reward
        # reward_normalize is determined by the config.
        cfg.is_collect = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        # when in evaluate phase, we don't need to normalize the reward.
        cfg.reward_normalize = False
        cfg.is_collect = False
        return [cfg for _ in range(evaluator_env_num)]
    
    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg
    
    def seed(self, seed=None, seed1=None):
        """Set the random seed for the gym environment."""
        self.np_random, seed = 0, 0
        return [seed]

