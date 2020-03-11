#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import signal
import argparse
import numpy as np
import json

# from scene_loader import THORDiscreteEnvironment
from utils.tools import SimpleImageViewer
import ai2thor.controller
import cv2

GRID_SIZE = 0.25

#
# Navigate the scene using your keyboard
#

def key_press(key, mod):

  global human_agent_action, human_wants_restart, stop_requested, take_picture, invert_view

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

  # View actions
  if key == 112: # p/P - Print view
    take_picture = True
  if key == 105: # i/I - Switch between RGB and Depth views
    invert_view = not invert_view


def rollout(event, controller, viewer, scene_name):

  global human_agent_action, human_wants_restart, stop_requested, take_picture, invert_view
  human_agent_action = None
  human_wants_restart = False
  stop_requested = False
  take_picture = False
  invert_view = False

  while True:
    # waiting for keyboard input
    if human_agent_action is not None:
      # move actions
      event = controller.step(action=human_agent_action)
      human_agent_action = None

    # waiting for reset command
    if human_wants_restart:
      # reset agent to random location
      controller.reset(scene=scene_name)
      human_wants_restart = False

    # check collision
    if event.metadata['collided']:
      print('Collision occurs.')
      event.collided = False

    # check quit command
    if stop_requested: break

    if take_picture:
      current_image = event.cv2image()
      cv2.imwrite("data/{}_goal.png".format(scene_name), current_image)
      json_dict = {}
      agent_position = event.metadata["agent"]["position"]
      agent_rotation = event.metadata['agent']['rotation']
      json_dict["grid_size"] = GRID_SIZE
      json_dict["agent_position"] = agent_position
      json_dict["agent_rotation"] = agent_rotation

      with open('data/{}_goal.json'.format(scene_name), 'w') as outfile:
        json.dump(json_dict, outfile)

    if invert_view and event.depth_frame is not None:
      depth_image = event.depth_frame
      # scale to be 0-255
      depth_image /= np.max(depth_image)
      depth_image *= 255
      viewer.imshow(depth_image.astype("uint8")) # cast as an int to ensure proper number of bytes
    else:
      viewer.imshow(event.frame)

if __name__ == '__main__':
  # scene_name = 'FloorPlan227'
  scene_name = 'FloorPlan402'

  controller = ai2thor.controller.Controller(scene=scene_name, gridSize=0.25, width=1000, height=1000, renderDepthImage=True);
  event = controller.step(action='Pass') # Perform no action

  human_agent_action = None
  human_wants_restart = False
  stop_requested = False
  take_picture = False
  invert_view = False

  viewer = SimpleImageViewer()
  viewer.imshow(event.frame)
  viewer.window.on_key_press = key_press

  print("Use WASD keys to move the agent.")
  print("Use QE keys to move the camera.")
  print("Press I to switch between RGB and Depth views.")
  print("Press P to save an image of the current view.")
  print("Press R to reset agent\'s location.")
  print("Press F to quit.")

  rollout(event, controller, viewer, scene_name)

  print("Goodbye.")
