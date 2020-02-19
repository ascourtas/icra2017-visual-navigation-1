#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ai2thor.controller
from agent_RL_controller import RLController

import tensorflow as tf
# NOTE: necessary for only using v1 code
import tensorflow.compat.v1 as tf
import numpy as np
import random
import sys
import json

from network import ActorCriticFFNetwork
from training_thread import A3CTrainingThread

from utils.ops import sample_action

from constants import ACTION_SIZE
from constants import CHECKPOINT_DIR
from constants import NUM_EVAL_EPISODES
from constants import VERBOSE

from constants import TASK_TYPE
from constants import TASK_LIST

GOAL_FILE = "data/FP227_goal_TV.json"
GOAL_POS = None

def main():
    # disable all v2 behavior
    tf.disable_v2_behavior()
    tf.disable_eager_execution()

    device = "/cpu:0"  # use CPU for display tool
    network_scope = TASK_TYPE # Always 'navigation'
    list_of_tasks = TASK_LIST
    scene_scopes = list_of_tasks.keys()

    global_network = ActorCriticFFNetwork(action_size=ACTION_SIZE,
                                          device=device,
                                          network_scope=network_scope,
                                          scene_scopes=scene_scopes)
    sess = tf.Session()
    # sess = tf.coSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

    # see if we saved a checkpoint from past training?
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint loaded: {}".format(checkpoint.model_checkpoint_path))
    else:
        print("Could not find old checkpoint")

    scene_stats = dict()
    for scene_scope in scene_scopes:
        scene_stats[scene_scope] = []
        for task_scope in list_of_tasks[scene_scope]:
            # tasks are positions!!!
            # env = ai2thor.controller.Controller(scene="FloorPlan227", gridSize=0.25, width=1000, height=1000)
            with open(GOAL_FILE, 'r') as f:
                GOAL_DATA = json.load(f)

            GOAL_POS = GOAL_DATA["agent_position"]
            env = RLController({
                'scene': scene_scope,
                'terminal_state_id': int(task_scope),
                'goal_pos': GOAL_POS,
                'goal_image_fpath': "data/FP227_goal_TV.png"
            })
            ep_rewards = []
            ep_lengths = []
            ep_collisions = []

            scopes = [network_scope, scene_scope]

            for i_episode in range(NUM_EVAL_EPISODES):
                env.reset()

                terminal = False
                ep_reward = 0
                ep_collision = 0
                ep_t = 0

                while not terminal:
                    # mirrors actions taken in paper
                    # NOTE: rearranged these to mirror code in scene_loader
                    list_of_actions = ["MoveAhead", "RotateRight", "RotateLeft", "MoveBack"]

                    pi_values = global_network.run_policy(sess, env.curr_state, env.target, scopes)
                    # action returned is an integer -- critical that the list_of_actions is in correct order
                    action = sample_action(pi_values)
                    print("Ep_t: {} Action: {}".format(ep_t, list_of_actions[action]))
                    env.step(list_of_actions[action])
                    env.update()

                    terminal = env.terminal
                    if ep_t == 10000: break

                    if env.collided: ep_collision += 1
                    ep_reward += env.reward
                    ep_t += 1
                print("we're done")

if __name__ == "__main__":
    main()