from __future__ import print_function
from matplotlib.widgets import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class CropImage(object):
    def __init__(self, image):
        self.image_coordinates = []
        fig, current_ax = plt.subplots()
        plt.imshow(image)

        toggle_selector.RS = RectangleSelector(current_ax, self.line_select_callback,
                                               drawtype='box', useblit=True,
                                               button=[1, 3],  # don't use middle button
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        plt.connect('key_press_event', toggle_selector)
        plt.show()

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))
        self.image_coordinates = [[int(x1), int(y1)],[int(x2), int(y2)]]
    def get_crop_coordinates(self):
        return self.image_coordinates
def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)

if __name__ == '__main__':
    image = Image.open('../../../1.jpg', 'r')
    ci = CropImage(np.asarray(image))
    print(ci.image_coordinates)
