import cv2
import numpy as np

class ActivationMap(object):
    def __init__(self):
        return

    def get_initial_map(self, img):
        return 1.0 - np.maximum(img[:,:,0], np.maximum(img[:,:,1], img[:,:,2]))

    def get_activation_map(self, img, dst):
        Wb = self.get_initial_map(img)
        varX_img = np.sqrt(np.abs(cv2.GaussianBlur(dst**2, (3, 3), sigmaX=2.0, sigmaY=0.0) - cv2.GaussianBlur(dst, (3, 3), sigmaX=2.0, sigmaY=0.0) ** 2) + 1e-3)
        varY_img = np.sqrt(np.abs(cv2.GaussianBlur(dst ** 2, (3, 3), sigmaX=0.0, sigmaY=2.0) - cv2.GaussianBlur(dst, (3, 3), sigmaX=0.0,sigmaY=2.0) ** 2) + 1e-3)
        Ax = 1.0 / (varX_img + 1e-3)
        Ay = 1.0 / (varY_img + 1e-3)
        activation_mapX = Wb * Ax
        activation_mapY = Wb * Ay
        #cv2.imshow("Activation Map", activation_map.astype(dtype=np.uint8))
        #cv2.waitKey(0)
        return activation_mapX, activation_mapY