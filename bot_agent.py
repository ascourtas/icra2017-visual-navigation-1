print("at the tippity top")

import ai2thor.controller
import pprint
import cv2
import numpy as np
import subprocess


# TODO: account for rotation as well (like with floor lamp)
def get_goal_position(goal_object_position, reachable_positions):
    """
    Find the reachable position closest to our goal object position

    :param goal_object_position:
    :param reachable_positions:
    :return final_pos:
    """
    final_pos = {}
    for coord_str in ['x', 'y', 'z']: # TODO: don't actually have to break them up, can just use whole positions
        # reset the min difference per coordinate
        min_pos_diff = np.inf
        for rp in reachable_positions:
            pos_diff = abs(goal_object_position[coord_str] - rp[coord_str])
            # if this difference is less than the min, update the min
            if pos_diff < min_pos_diff:
                # update the coordinate to with the closer coord
                final_pos[coord_str] = rp[coord_str]
                min_pos_diff = pos_diff

    return final_pos

print("Before controller")
# subprocess.call(['startx'])
process = subprocess.Popen(['xdpyinfo'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

stdout = process.communicate()[0]
print('STDOUT:{}'.format(stdout))
# the floor plan names correspond to those in constants.py
controller = ai2thor.controller.Controller(scene='FloorPlan227', gridSize=0.25, width=1000, height=1000)
# controller.docker_enabled = True

# # TV position is {'x': -3.503, 'y': 1.506, 'z': 0.001}
goal_object = {"name": "TV", "position": {'x': -3.503, 'y': 1.506, 'z': 0.001}}
# # Floor lamp position is {'x': -0.369, 'y': 0.026053369, 'z': 4.992}, rotation: {'x': 0.0, 'y': 59.9998856, 'z': 0.0}
# goal_object_position = {"name": "FloorLamp", "position": {'x': -0.369, 'y': 0.026053369, 'z': 4.992}}

# get the positions that the agent could possibly reach
event = controller.step(action='GetReachablePositions')
reachable_positions = event.metadata['reachablePositions']

# get the reachable position closest to our goal object
final_pos = get_goal_position(goal_object['position'], reachable_positions)
event = controller.step(dict(action='Teleport', x=final_pos['x'], y=final_pos['y'], z=final_pos['z']))

# # Numpy Array - shape (width, height, channels), channels are in RGB order
# current_image = event.frame

# # Numpy Array in BGR order suitable for use with OpenCV
current_image = event.cv2image()
cv2.imwrite("data/FP227_goal_{}.png".format(goal_object['name']), current_image)

# current metadata dictionary that includes the state of the scene
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(event.metadata)