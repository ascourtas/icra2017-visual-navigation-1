import ai2thor.controller
import pprint

controller = ai2thor.controller.Controller(scene='FloorPlan28', gridSize=0.25, width=1000, height=1000)
print("before controller")
event = controller.step(dict(action='MoveAhead'))
print("bob")
# Numpy Array - shape (width, height, channels), channels are in RGB order
event.frame
print("susan")
# Numpy Array in BGR order suitable for use with OpenCV
image = event.cv2image()
print("karen")
# current metadata dictionary that includes the state of the scene
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(event.metadata)