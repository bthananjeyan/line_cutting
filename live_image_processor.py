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
from image_processor import *

class ImageRecorder:

    def __init__(self):
        self.right_image = None
        self.left_image = None
        self.info = {'l': None, 'r': None}
        self.bridge = cv_bridge.CvBridge()
        self.error_detector = ErrorDetector()


        #========SUBSCRIBERS========#
        # image subscribers
        rospy.init_node('image_saver')
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
        self.right_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")


    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.error_detector.no_error(self.left_image)



if __name__ == "__main__":
    a = ImageRecorder()

