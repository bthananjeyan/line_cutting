import rospy
import cv
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
import cv2
import cv_bridge
import numpy as np
import rospy, scipy.misc

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

import matplotlib.pyplot as plt

class LineDetector:

    def __init__(self, USE_SAVED_IMAGES = True):
        self.right_image = None
        self.left_image = None
        self.info = {'l': None, 'r': None}
        self.USE_SAVED_IMAGES = USE_SAVED_IMAGES
        self.bridge = cv_bridge.CvBridge()

        if not USE_SAVED_IMAGES:
            #========SUBSCRIBERS========#
            # image subscribers
            rospy.init_node('circle_detector')
            rospy.Subscriber("/endoscope/left/image_color", Image,
                             self.left_image_callback, queue_size=1)
            rospy.Subscriber("/endoscope/right/image_color", Image,
                             self.right_image_callback, queue_size=1)
            # info subscribers
            rospy.Subscriber("/endoscope/left/camera_info",
                             CameraInfo, self.left_info_callback)
            rospy.Subscriber("/endoscope/right/camera_info",
                             CameraInfo, self.right_info_callback)
            rospy.spin()
        else:
            self.right_image = cv2.imread('images/right_checkerboard.jpg')
            self.left_image = cv2.imread('images/left_checkerboard.jpg')

    def left_info_callback(self, msg):
        if self.info['l']:
            return
        self.info['l'] = msg

    def right_info_callback(self, msg):
        if self.info['r']:
            return
        self.info['r'] = msg

    def right_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        if self.USE_SAVED_IMAGES:
            self.right_image = cv2.imread('images/right_checkerboard.jpg')
        elif self.right_image != None:
            return
        else:
            self.right_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            scipy.misc.imsave('images/right_checkerboard.jpg', self.right_image)


    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        if self.USE_SAVED_IMAGES:
            self.left_image = cv2.imread('images/left_checkerboard.jpg')
        elif self.left_image != None:
            pass
        else:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            scipy.misc.imsave('images/left_checkerboard.jpg', self.left_image)
        if self.right_image != None:
            self.process_image()

    def process_image(self):
        print True
        self.left_image = self.right_image

        hsv = cv2.cvtColor(self.left_image, cv2.COLOR_BGR2HSV)
        frame = self.left_image

        # define range of blue color in HSV
        lower_blue = np.array([110,50,0])
        upper_blue = np.array([130,255,255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)
        res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        res = cv2.adaptiveThreshold(res,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,101,2)
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
        res = cv2.dilate(res, dilation_kernel)
        res = cv2.dilate(res, dilation_kernel)
        plt.imshow(res)
        plt.show()
        scipy.misc.imsave('asdf.jpg', res)
        return
        left_gray = cv2.cvtColor(self.left_image,cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(self.right_image,cv2.COLOR_BGR2GRAY)

        thresholded_image = cv2.adaptiveThreshold(left_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,381,2)
        kernel = np.ones((25,25),np.float32)/625
        thresholded_image = cv2.filter2D(thresholded_image,-1,kernel)
        ret, thresholded_image = cv2.threshold(thresholded_image, 120, 255, cv2.THRESH_BINARY_INV)

        # thresholded_image = thresholded_image * -1 + 255

        # dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        # thresholded_image = cv2.dilate(thresholded_image, dilation_kernel)
        # thresholded_image = cv2.dilate(thresholded_image, dilation_kernel)
        # th3 = cv2.adaptiveThreshold(self.left_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        # cv2.THRESH_BINARY,11,2)

        contours, contour_hierarchy = cv2.findContours(thresholded_image.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        print len(contours)


        plt.imshow(thresholded_image, cmap='Greys_r')
        plt.show()
        # threshold = 120
        scipy.misc.imsave('threshold.jpg', thresholded_image)


        # thresholded_image = cv2.dilate(thresholded_image, dilation_kernel)
        # scipy.misc.imsave('dilated.jpg', thresholded_image)
 


        return


if __name__ == "__main__":
    a = LineDetector(USE_SAVED_IMAGES=True)
    a.process_image()
