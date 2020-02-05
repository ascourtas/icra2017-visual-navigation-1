#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ai2thor.controller

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
GOAL_STATE = None

def main():
    # disable all v2 behavior
    tf.disable_v2_behavior()
    tf.disable_eager_execution()

    device = "/cpu:0"  # use CPU for display tool
    network_scope = TASK_TYPE
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
        # tasks are positions!!!
        env = ai2thor.controller.Controller(scene="FloorPlan227", gridSize=0.25, width=1000, height=1000)
        ep_rewards = []
        ep_lengths = []
        ep_collisions = []
        with open(GOAL_FILE, 'r') as f:
            GOAL_DATA = json.load(f)

        GOAL_STATE = GOAL_DATA["agent_position"]
        scopes = [network_scope, scene_scope]

        for i_episode in range(NUM_EVAL_EPISODES):
            #
            # env.reset(scene=)
            # TODO: pull out reward code from old reset()
            terminal = False
            ep_reward = 0
            ep_collision = 0
            ep_t = 0

            while not terminal and ep_t != 10000:
                # TODO: add remaining actions
                # TODO: randomly choose from these actions, don't just loop through them
                action = random.choice(["RotateRight", "RotateLeft", "MoveAhead"])
                # TODO: implement new policy function

                # # NOTE: old action choosing code
                # pi_values = global_network.run_policy(sess, env.s_t, env.target, scopes)
                # action = sample_action(pi_values)

                # TODO: pick targets/terminal states (need the target image)
                # TODO: go through old code and figure out how to check for terminal state image
                event = env.step(action)
                # TODO: update the state
                # env.update()    # # evaluates to self.s_t = self.s_t1
                if event.metadata["agent"]["position"] == GOAL_STATE:
                    terminal = True

                # TODO: check for terminal position
                # terminal = env.terminal
                if terminal or ep_t == 10000:
                    break
                # if ep_t == 10000: break
                # TODO: go through old code and figure out how to check that we've collided
                # if env.collided: ep_collision += 1
                # TODO: move all reward-related code from old code to here
                # ep_reward += env.reward
                ep_t += 1
    # while True:


if __name__ == "__main__":
    main()