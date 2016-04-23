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
            rospy.Subscriber("/endoscope/left/image_raw", Image,
                             self.left_image_callback, queue_size=1)
            rospy.Subscriber("/endoscope/right/image_raw", Image,
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
            self.right_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            scipy.misc.imsave('images/right_checkerboard.jpg', self.right_image)


    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        if self.USE_SAVED_IMAGES:
            self.left_image = cv2.imread('images/left_checkerboard.jpg')
        elif self.left_image != None:
            pass
        else:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            scipy.misc.imsave('images/left_checkerboard.jpg', self.left_image)
        if self.right_image != None:
            self.process_image()

    def process_image(self):
        print True
        return


if __name__ == "__main__":
    a = LineDetector(USE_SAVED_IMAGES=False)
