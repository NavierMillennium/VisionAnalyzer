import os, sys
from camera import *

logging.basicConfig(level=logging.DEBUG)
# Project path
PATH = r'\path_22_08'
FOLDER_NAME = 'usb_calib_photo'
JSON_NAME = 'usb_conf'
dim = (7,8)
VideoAnalyzer.calibrate_camera(dim, PATH, FOLDER_NAME, JSON_NAME, square_size = 30/1000)