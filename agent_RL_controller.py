import ai2thor.controller
import numpy as np
import random
import h5py
from constants import ACTION_SIZE
from constants import SCREEN_WIDTH
from constants import SCREEN_HEIGHT
from constants import HISTORY_LENGTH

class RLController :
    def __init__(self, config=dict()):
        self.scene          = config.get('scene', 'bedroom_04')
        self.random_start        = config.get('random_start', True)
        self.n_feat_per_locaiton = config.get('n_feat_per_locaiton', 1) # 1 for no sampling
        self.terminal_state_id   = config.get('terminal_state_id', 0)
        self.goal_pos            = config.get('goal_pos', {'x': 0, 'y': 0, 'z': 0})

        self.h5_file_path = config.get('h5_file_path', 'data/%s.h5' % self.scene)
        self.h5_file = h5py.File(self.h5_file_path, 'r')

        self.env = ai2thor.controller.Controller(scene=self.scene, gridSize=0.25, width=1000, height=1000)
        self.reward = 0
        self.collided = False
        self.terminal = False

        # TODO: Import constants.py
        self.history_length = HISTORY_LENGTH
        self.screen_height = SCREEN_HEIGHT
        self.screen_width = SCREEN_WIDTH

        # TODO: Figure out how to incorporate into training thread
        # we use pre-computed fc7 features from ResNet-50
        ## self.curr_state = np.zeros([self.screen_height, self.screen_width, self.history_length])
        self.curr_state = np.zeros([2048, self.history_length])
        self.next_state = np.zeros_like(self.curr_state)
        self.s_target = self._tiled_state(self.terminal_state_id)

    def reset(self):
        # TODO: Check if goal state is reachable
        self.env.reset(self.scene)
        self.reward = 0
        self.collided = False
        self.terminal = False

    def step(self, action):
        event = self.env.step(action=action)

        assert not self.terminal, 'step() called in terminal state'
        curr_pos = event.metadata["agent"]["position"]

        # TODO: this will never be reached in current code due to goal position not being in all scenes;
        #  need to figure out key id system in OG code so we can either adopt it or can it
        if curr_pos == self.goal_pos:
            self.terminal = True

        self.collided = event.metadata["collided"]
        self.reward = self._reward()

        # TODO: Make this work so we can actually learn
        # self.next_state = np.append(self.curr_state[:, 1:], self.state, axis=1)

    def update(self):
        # TODO: Make this work so we can actually learn our policy
        # self.curr_state = self.next_state
        pass

    def _tiled_state(self, state_id):
        k = random.randrange(self.n_feat_per_locaiton)
        f = self.h5_file['resnet_feature'][state_id][k][:, np.newaxis]  # f is some portion of the RestNet features
        return np.tile(f, (1, self.history_length))  # Repeats f (1, self.history_length) times

    def _reward(self):
        # positive reward upon task completion
        if self.terminal: return 10.0
        # time penalty or collision penalty
        return -0.1 if self.collided else -0.01

    @property
    def target(self):
        return self.s_target

    # @property
    # def state(self):
    #     # read from hdf5 cache
    #     k = random.randrange(self.n_feat_per_locaiton)
    #     return self.h5_file['resnet_feature'][self.current_state_id][k][:, np.newaxis]