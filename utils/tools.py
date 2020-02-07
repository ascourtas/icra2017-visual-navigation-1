import pyglet

class SimpleImageViewer(object):

  def __init__(self, display=None):
    self.window = None
    self.isopen = False
    self.display = display

  def imshow(self, arr):
    if self.window is None:
      height, width, channels = arr.shape
      self.window = pyglet.window.Window(width=width, height=height, display=self.display, caption="THOR Browser")
      self.width = width
      self.height = height
      self.isopen = True

    #assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
    # TODO: cleanup
    assert arr.shape[0] == self.height
    assert arr.shape[1] == self.width
    # print("Shape is {}".format(arr.shape))
    if len(arr.shape) == 2:
      # TODO: the pitch should be equal to the number of bytes in each row -- these are floats, is that the issue?
      image = pyglet.image.ImageData(self.width, self.height, 'I', arr.tobytes(), pitch=self.width * -1)
    elif arr.shape[2] == 1:
      image = pyglet.image.ImageData(self.width, self.height, 'I', arr.tobytes(), pitch=self.width * -1)
    elif arr.shape[2] == 3:
      image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
    else:
      assert False, "Number of channels passed is not supported."
    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    image.blit(0,0)
    self.window.flip()

  def close(self):
    if self.isopen:
      self.window.close()
      self.isopen = False

  def __del__(self):
    self.close()
