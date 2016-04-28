import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from PIL import *
import scipy.ndimage
import math

class ErrorDetector:

    def findGreen(self, image):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        frame = image

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
        # print np.sum(res)
        return res

    def center_of_mass(self, image):
        return scipy.ndimage.measurements.center_of_mass(image)

    def error_threshold(self, image, threshold):
        res = self.findGreen(im)
        res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        res = 255 - res
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
        res = cv2.dilate(res, dilation_kernel)
        res = cv2.dilate(res, dilation_kernel)
        res = 255 - res
        return np.sum(res) < threshold

    def find_line(self, image):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        frame = im
        lower_blue = np.array([0,50,30])
        upper_blue = np.array([30,255,255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(frame,frame, mask= mask)
        res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        res = 255 - res
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
        res = cv2.dilate(res, dilation_kernel)
        res = cv2.dilate(res, dilation_kernel)
        res = 255 - res
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
        res = cv2.dilate(res, dilation_kernel)
        res = cv2.dilate(res, dilation_kernel)
        threshold = 60
        ret, res = cv2.threshold(res, threshold, 255, cv2.THRESH_BINARY)
        y, x = np.nonzero(res)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y)[0]

        # x = np.r_[0:1700]
        # y = m * x + c
        

        # contours, contour_hierarchy = cv2.findContours(res.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # areas = [cv2.contourArea(i) for i in contours]
        # for area in areas:
        #     if area > 1000:
        #         print area
        # plt.imshow(res, cmap='Greys_r')
        # plt.plot(x, y, "o")
        # plt.scatter(pt[0], pt[1], s = 10, c = 1)
        # plt.show()
        return m, c

    def distance_to_line(self, pt, line):
        m, c = line
        x0, y0 = pt
        numerator = float(m * x0 - y0 + c)
        denominator = math.sqrt(float(m **2 + 1))

        return numerator / denominator

    def distance_to_line_image(self, image):
        im1 = self.no_error(image)
        y, x = self.center_of_mass(im1)
        pt = (x, y)
        line = self.find_line(image)
        return self.distance_to_line(pt, line)

if __name__ == "__main__":

    e = ErrorDetector()
    for x in range(0, 76):
        im = cv2.imread('images/left' + str(x) + '.jpg')
        print x
        # res, err = e.no_error(im)
        # print np.sum(res)
        # plt.imshow(res, cmap='Greys_r')
        # plt.show()
        # if not err:
        #     y, x = e.center_of_mass(res)
        #     pt = (x, y)
        #     line =  e.find_line(im)
        #     print e.distance_to_line(pt, line)
        print e.distance_to_line_image(im)