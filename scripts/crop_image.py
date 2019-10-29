import cv2
import numpy as np


class CropImage(object):
    def __init__(self, cv_image):
        self.original_image = cv_image

        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # Bounding box reference points and boolean if we are extracting coordinates
        self.image_coordinates = []
        self.extract = False

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]
            self.extract = True

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            self.extract = False
            print('top left: {}, bottom right: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))

            # Draw rectangle around ROI
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (0,255,0), 2)
            cv2.imshow("image", self.clone)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone
    def get_crop_coordinates(self):
        return self.image_coordinates


if __name__ == '__main__':
    image = cv2.imread("../data/real_aspects/Aspect-Raw1.jpg")
    ci = CropImage(image)

    cv2.imshow('image', ci.show_image())
    key = cv2.waitKey(0)
    print(ci.get_crop_coordinates())
