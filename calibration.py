import cv2
import Tkinter
from Tkinter import *
import rospy, pickle, time
from geometry_msgs.msg import Pose
import multiprocessing
import numpy as np
import sys
from sklearn.decomposition import PCA


def startCallback():
    global prs
    process = multiprocessing.Process(target = start_listening)
    prs.append(process)
    process.start()
    return

def start_listening():
    global sub
    rospy.init_node('listener', anonymous=True)
    sub = rospy.Subscriber('/dvrk/PSM1/position_cartesian_current', Pose, callback_PSM1_actual)
    rospy.spin()

def exitCallback():
    global prs
    for process in prs:
        process.terminate()
    sys.exit()

def callback_PSM1_actual(data):
    global psm1_pose
    position = data.position
    psm1_pose = [position.x, position.y, position.z]
    sub.unregister()
    f = open("calibration/psm1_calibration.p", "a")
    pickle.dump(psm1_pose, f)
    f.close()

if __name__ == '__main__':
    sub = None
    prs = []
    psm1_pose = None
    points = []

    open('calibration/psm1_calibration.p', 'w+').close()

    top = Tkinter.Tk()
    top.title('Calibration')
    top.geometry('400x200')

    B = Tkinter.Button(top, text="Record Position", command = startCallback)
    D = Tkinter.Button(top, text="Exit", command = exitCallback)

    B.pack()
    D.pack()

    top.mainloop()
