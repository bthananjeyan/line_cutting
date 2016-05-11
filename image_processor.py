import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from PIL import *
import scipy.ndimage
import math

import image_geometry
import rospy
import cv
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
import cv_bridge
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped

import solve_system


def convertStereo(u, v, disparity, info):
    """
    Converts two pixel coordinates u and v along with the disparity to give PointStamped       
    """
    stereoModel = image_geometry.StereoCameraModel()
    stereoModel.fromCameraInfo(info['l'], info['r'])
    (x,y,z) = stereoModel.projectPixelTo3d((u,v), disparity)

    cameraPoint = PointStamped()
    cameraPoint.header.frame_id = info['l'].header.frame_id
    cameraPoint.header.stamp = rospy.Time.now()
    cameraPoint.point = Point(x,y,z)
    return cameraPoint

class ImageProcessor:

    def findGreen(self, image):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        frame = image

        lower_blue = np.array([60,80,70])
        upper_blue = np.array([90,255,255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        res = cv2.bitwise_and(frame,frame, mask= mask)

        res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        res = 255 - res
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
        res = cv2.dilate(res, dilation_kernel)
        res = cv2.dilate(res, dilation_kernel)
        res = 255 - res

        return self.center_of_mass(res)

    def center_of_mass(self, image):
        return scipy.ndimage.measurements.center_of_mass(image)

    def find_line(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        frame = image
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

        threshold = 20
        ret, res = cv2.threshold(res, threshold, 255, cv2.THRESH_BINARY)
        # plt.imshow(res, cmap='Greys_r')
        # plt.show()
        # y, x = np.nonzero(res)
        # A = np.vstack([x, np.ones(len(x))]).T
        # m, c = np.linalg.lstsq(A, y)[0]
        # yunit = np.around((max(y) - min(y)) / 100.0)
        # xunit = np.around((max(x) - min(x)) / 100.0)

        # x = np.r_[0:1700]
        # y = m * x + c


        # contours, contour_hierarchy = cv2.findContours(res.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        # areas = [cv2.contourArea(i) for i in contours]
        # idx = None
        # for i in range(len(areas)):
        #     if areas[i] > 1000:
        #         idx = i
        # pts = contours[i].reshape(contours[i].shape[0], contours[i].shape[2])
        # maxright = np.argmax(pts, axis = 0)
        # maxleft = np.argmin(pts, axis = 0)
        # print contours[i]
        # for area in areas:
        #     if area > 1000:
        #         print area
        # plt.imshow(res, cmap='Greys_r')
        # plt.plot(x, y, "o")
        # plt.scatter(pt[0], pt[1], s = 10, c = 1)
        # plt.show()
        return res

    def bounding_box(self, image, size = 100):
        center = self.findGreen(image)
        nw = (center[1] - size, center[0] - size)
        ne = (center[1] + size, center[0] - size)
        sw = (center[1] - size, center[0] + size)
        se = (center[1] + size, center[0] + size)
        return (nw, ne, sw, se)

    def is_valid_pt(self, pt, box):
        if not (pt[0] > box[0][0] and pt[1] > box[0][1]):
            return False
        elif not (pt[0] < box[1][0] and pt[1] > box[1][1]):
            return False
        elif not (pt[0] > box[2][0] and pt[1] < box[2][1]):
            return False
        elif not (pt[0] < box[3][0] and pt[1] < box[3][1]):
            return False
        else:
            return True

    def find_line_bounded(self, image):
        box = self.bounding_box(image, size = 60)
        res = self.find_line(image)
        y, x = np.nonzero(res)
        pts = np.vstack((x, y)).T
        valid_pts = []
        for i in range(pts.shape[0]):
            pt = pts[i, :]
            if not self.is_valid_pt(pt, box):
                continue
            valid_pts.append(pt)
        x = np.array([pt[0] for pt in valid_pts])
        y = np.array([pt[1] for pt in valid_pts])
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y)[0]

        # x = np.r_[0:1700]
        # y = m * x + c
        # plt.imshow(res, cmap='Greys_r')
        # plt.plot(x, y, "o")
        # plt.show()

        x = np.r_[np.around(box[0][0]): np.around(box[1][0]):10]
        y = m * x + c


        return np.vstack((np.around(y), x)).T

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

class LineSubscriber:

    USE_SAVED_IMAGES = False

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.left_image = None
        self.right_image = None
        self.info = {'l': None, 'r': None, 'b': None, 'd': None}
        self.imp = ImageProcessor()
        self.camera_mat = solve_system.solve_for_camera_matrix()
        self.robot_mat = solve_system.solve_for_robot_matrix()

        #========SUBSCRIBERS========#
        # image subscribers
        rospy.Subscriber("/endoscope/left/image_rect_color", Image,
                         self.left_image_callback, queue_size=1)
        rospy.Subscriber("/endoscope/right/image_rect_color", Image,
                         self.right_image_callback, queue_size=1)
        # info subscribers
        rospy.Subscriber("/endoscope/left/camera_info",
                         CameraInfo, self.left_info_callback)
        rospy.Subscriber("/endoscope/right/camera_info",
                         CameraInfo, self.right_info_callback)

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
        if USE_SAVED_IMAGES:
            self.right_image = cv2.imread('right.jpg')
        else:
            self.right_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")


    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        if USE_SAVED_IMAGES:
            self.left_image = cv2.imread('left.jpg')
        else:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        if self.right_image != None:
            self.process_image()

    def get_points_3d(self, left_points, right_points):
        """ this method assumes that corresponding points are in the right order
            and returns a list of 3d points """

        # both lists must be of the same length otherwise return None
        if len(left_points) != len(right_points):
            rospy.logerror("The number of left points and the number of right points is not the same")
            return None

        points_3d = []
        for i in range(len(left_points)):
            a = left_points[i]
            b = right_points[i]
            disparity = abs(a[0]-b[0])
            pt = convertStereo(a[0], a[1], disparity, self.info)
            points_3d.append(pt)
        return points_3d
    
    def process_image(self):
        print "processing image"
        pts_left = self.imp.find_line_bounded(self.left_image)[:11]
        pts_right = self.imp.find_line_bounded(self.right_image)[:11]
        pts3d = self.get_points_3d(pts_left, pts_right)
        return pts3d

if __name__ == "__main__":

    e = ImageProcessor()
    im = cv2.imread('left.jpg')
    left_pts = e.find_line_bounded(im)
    print left_pts.shape
