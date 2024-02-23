from camera import *

logging.basicConfig(level=logging.DEBUG)
# Project path
PATH = r'\path_22_08'
JSON_POSE = 'pose_all'
# Create objcet for usb camera
usb = USBCamera(2)
# Waiting for camera initialization
time.sleep(2)

mk = VideoAnalyzer(PATH, usb)
# Number of corners vertical and horizontal
dim = (7,8)

mk.calib_image_saver(dim,'usb_calib_photo','usb_conf')
