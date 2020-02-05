import ai2thor.controller
import numpy as np

class RLController :
    def __init__(self, scene, goal_pos):
        self.scene = scene
        self.goal_pos = goal_pos
        self.env = ai2thor.controller.Controller(scene=scene, gridSize=0.25, width=1000, height=1000)
        self.reward = 0
        self.collided = False
        self.terminal = False

        # TODO: Import constants.py
        # self.history_length = HISTORY_LENGTH
        # self.screen_height = SCREEN_HEIGHT
        # self.screen_width = SCREEN_WIDTH

        # TODO: Figure out how to incorporate into training thread
        # we use pre-computed fc7 features from ResNet-50
        ## self.curr_state = np.zeros([self.screen_height, self.screen_width, self.history_length])
        # self.curr_state = np.zeros([2048, self.history_length])
        # self.next_state = np.zeros_like(self.s_t)
        # self.s_target = self._tiled_state(self.terminal_state_id)

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

        if curr_pos == self.goal_pos:
            self.terminal = True

        # TODO: Implement collision code
        # if self.terminals[self.current_state_id]:
        #     self.terminal = True
        #     self.collided = False
        # else:
        #     self.terminal = False
        #     self.collided = False
        # else:
        #     self.terminal = False
        #     self.collided = True

        self.reward = self._reward()

        # TODO: Integrate with training thread
        # self.next_state = np.append(self.curr_state[:, 1:], self.state, axis=1)

    def update(self):
        # TODO: Integrate with training thread
        # self.curr_state = self.next_state
        pass

    def _reward(self):
        # positive reward upon task completion
        if self.terminal: return 10.0
        # time penalty or collision penalty
        return -0.1 if self.collided else -0.01