#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import signal
import argparse
import numpy as np

# from scene_loader import THORDiscreteEnvironment
from utils.tools import SimpleImageViewer
import ai2thor.controller

#
# Navigate the scene using your keyboard
#

def key_press(key, mod):

  global human_agent_action, human_wants_restart, stop_requested

  # Browser actions
  if key == ord('R') or key == ord('r'): # r/R - Restart
    human_wants_restart = True
  if key == ord('F') or key == ord('f'): # f/F - Quit
    stop_requested = True

  # Agent actions
  if key == 119 or key == 0xFF52: # w/W - Up
    human_agent_action = "MoveAhead"
  if key == 100 or key == 0xFF53: # d/D - Right
    human_agent_action = "MoveRight"
  if key == 97 or key == 0xFF51: # a/A - Left
    human_agent_action = "MoveLeft"
  if key == 115 or key == 0xFF54: # s/S - Down
    human_agent_action = "MoveBack"

  # Camera actions
  if key == 101: # e/E - Rotate Camera Right
    human_agent_action = "RotateRight"
  if key == 113: # q/Q - Rotate Camera Left
    human_agent_action = "RotateLeft"

def rollout(event, controller, viewer):

  global human_agent_action, human_wants_restart, stop_requested
  human_agent_action = None
  human_wants_restart = False
  while True:
    # waiting for keyboard input
    if human_agent_action is not None:
      # move actions
      event = controller.step(action=human_agent_action)
      human_agent_action = None

    # waiting for reset command
    if human_wants_restart:
      # reset agent to random location
      controller.reset(scene='FloorPlan227')
      human_wants_restart = False

    # check collision
    if event.metadata['collided']:
      print('Collision occurs.')
      event.collided = False

    # check quit command
    if stop_requested: break

    viewer.imshow(event.frame)

if __name__ == '__main__':

  controller = ai2thor.controller.Controller(scene='FloorPlan227', gridSize=0.25, width=1000, height=1000);
  event = controller.step(action='Pass') # Perform no action

  human_agent_action = None
  human_wants_restart = False
  stop_requested = False

  viewer = SimpleImageViewer()
  viewer.imshow(event.frame)
  viewer.window.on_key_press = key_press

  print("Use WASD keys to move the agent.")
  print("Use QE keys to move the camera.")
  print("Press R to reset agent\'s location.")
  print("Press F to quit.")

  rollout(event, controller, viewer)

  print("Goodbye.")
