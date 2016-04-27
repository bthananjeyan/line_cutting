import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from PIL import *

class ErrorDetector:

    def findGreen(self, image):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        frame = im

        # define range of blue color in HSV
        lower_blue = np.array([60,80,70])
        upper_blue = np.array([90,255,255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)
        return res

    def no_error(self, image):
        res = self.findGreen(im)
        res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        res = 255 - res
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
        res = cv2.dilate(res, dilation_kernel)
        res = cv2.dilate(res, dilation_kernel)
        res = 255 - res
        print np.sum(res) < 150000
        return res

    def error_threshold(self, image, threshold):
        res = self.findGreen(im)
        res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        res = 255 - res
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
        res = cv2.dilate(res, dilation_kernel)
        res = cv2.dilate(res, dilation_kernel)
        res = 255 - res
        return np.sum(res) < threshold

if __name__ == "__main__":

    e = ErrorDetector()
    for x in range(0, 76):
        im = cv2.imread('images/left' + str(x) + '.jpg')
        print x
        res = e.no_error(im)
        # contours, contour_hierarchy = cv2.findContours(res.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # print len(contours)
        # for i in range(len(contours)):
        #     print cv2.contourArea(contours[i])
        # print np.sum(res)
        # plt.imshow(res, cmap='Greys_r')
        # plt.show()