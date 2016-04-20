import rospy, pickle, time
from robot import *
from geometry_msgs.msg import Pose
import numpy as np
import PyKDL
from scipy.interpolate import UnivariateSpline
import multiprocessing
import Tkinter
import tfx
import IPython


def initialize():
    global psm1_position, psm2_position
    psm2_initial = [-0.0912484814441, 0.0674414679811, -0.0736081506167, 0.888720223413, -0.254590620512, -0.302364852004, -0.232240127275]
    psm1_initial = [0.0298111217581, 0.0255537141169, -0.111452040502, 0.617571885272, 0.59489495214, 0.472153066551, 0.204392867261]
    start_frame1 = get_frame(psm1_initial)
    start_frame2 = get_frame(psm2_initial)
    psm1.move_cartesian_frame(start_frame1)
    psm2.move_cartesian_frame(start_frame2)
    psm1.open_gripper(80)
    psm2.open_gripper(80)
    psm1_position = psm1_initial
    psm2_position = psm2_initial
    time.sleep(2)
    return

def get_frame(pos):
    return tfx.pose(pos[0:3], pos[3:7])

def cut():
    psm1.open_gripper(-30)
    time.sleep(2.5)
    pos = psm1_position
    pos[0] = pos[0] - 0.0005
    psm1.open_gripper(80)
    time.sleep(2.35)
    psm1.move_cartesian_frame(get_frame(pos))

def move_to_next():
    pos = psm1_position
    pos[0] = pos[0] + 0.005
    pos[2] = pos[2] - 0.0004
    psm1.move_cartesian_frame(get_frame(pos))

def grab_gauze():
    pos = psm2_position
    pos[2] = pos[2] - 0.013
    psm2.move_cartesian_frame(get_frame(pos))
    psm2.open_gripper(-30)
    time.sleep(2.5)
    pos[2] = pos[2] + 0.01
    psm2.move_cartesian_frame(get_frame(pos))


if __name__ == '__main__':
    psm1_position = None
    psm2_position = None

    psm1 = robot("PSM1")
    psm2 = robot("PSM2")

    initialize()

    grab_gauze()

    for i in range(25):
        cut()
        move_to_next()
